import json
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Final
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TypedDict
from typing import Union

import numpy as np
from einops import rearrange
from loguru import logger
from numpy.typing import NDArray

from avalon.agent.godot.godot_gym import create_base_benchmark_config
from avalon.common.utils import wait_until_true
from avalon.common.visual_utils import visualize_arraylike_as_video
from avalon.datagen.godot_env.actions import AttrsAction
from avalon.datagen.godot_env.actions import MouseKeyboardAction
from avalon.datagen.godot_env.actions import VRAction
from avalon.datagen.godot_env.goals import NullGoalEvaluator
from avalon.datagen.godot_env.godot_env import GodotEnv
from avalon.datagen.godot_env.observations import AttrsObservation
from avalon.datagen.godot_env.observations import AvalonObservation
from avalon.datagen.godot_env.replay import get_action_type_from_config as _get_action_type_from_config
from avalon.datagen.godot_generated_types import AgentPlayerSpec
from avalon.datagen.godot_generated_types import AvalonSimSpec
from avalon.datagen.godot_generated_types import MouseKeyboardAgentPlayerSpec
from avalon.datagen.godot_generated_types import VRAgentPlayerSpec
from avalon.datagen.godot_utils import S3_AVALON_ERROR_BUCKET

_ActOrStepObservationSequence = Union[Sequence[AvalonObservation], Sequence[Dict[str, Any]]]

get_action_type_from_config: Final = _get_action_type_from_config


def create_mouse_keyboard_benchmark_config() -> AvalonSimSpec:
    with create_base_benchmark_config().mutable_clone() as config:
        assert isinstance(config.player, AgentPlayerSpec)
        config.player = MouseKeyboardAgentPlayerSpec.from_dict(config.player.to_dict())
        return config


def create_vr_benchmark_config() -> AvalonSimSpec:
    with create_base_benchmark_config().mutable_clone() as config:
        assert isinstance(config.player, AgentPlayerSpec)
        config.player = VRAgentPlayerSpec.from_dict(config.player.to_dict())
        return config


def get_null_mouse_keyboard_action() -> AttrsAction:
    return MouseKeyboardAction(
        head_x=0.0,
        head_z=0.0,
        head_pitch=0.0,
        head_yaw=0.0,
        is_left_hand_grasping=0.0,
        is_right_hand_grasping=0.0,
        is_left_hand_throwing=0.0,
        is_right_hand_throwing=0.0,
        is_jumping=0.0,
        is_eating=0.0,
        is_crouching=0.0,
    )


def get_null_vr_action() -> VRAction:
    return VRAction(
        head_x=0.0,
        head_y=0.0,
        head_z=0.0,
        head_pitch=0.0,
        head_yaw=0.0,
        head_roll=0.0,
        left_hand_x=0.0,
        left_hand_y=0.0,
        left_hand_z=0.0,
        left_hand_pitch=0.0,
        left_hand_yaw=0.0,
        left_hand_roll=0.0,
        is_left_hand_grasping=0.0,
        right_hand_x=0.0,
        right_hand_y=0.0,
        right_hand_z=0.0,
        right_hand_pitch=0.0,
        right_hand_yaw=0.0,
        right_hand_roll=0.0,
        is_right_hand_grasping=0.0,
        is_jumping=0.0,
    )


def create_env(
    config: AvalonSimSpec,
    action_type: Type[AttrsAction],
    is_logging_artifacts_on_error_to_s3: bool = False,
    observation_type: Type[AttrsObservation] = AvalonObservation,
) -> GodotEnv:
    return GodotEnv(
        config=config,
        observation_type=observation_type,
        action_type=action_type,
        goal_evaluator=NullGoalEvaluator(),
        is_logging_artifacts_on_error_to_s3=is_logging_artifacts_on_error_to_s3,
        s3_bucket_name=S3_AVALON_ERROR_BUCKET,
    )


def rgbd_to_video_array(rgbd_data: Iterable[NDArray]) -> NDArray:
    return np.stack(
        [
            rearrange(
                np.flipud(rgbd[:, :, :3]),
                "h w c -> c h w",
            )
            / 255.0
            for rgbd in rgbd_data
        ]
    )


def observation_video_array(data: _ActOrStepObservationSequence) -> NDArray:
    if len(data) == 0:
        return rgbd_to_video_array([])

    if isinstance(data[0], dict):
        assert "rgbd" in data[0], "observation dict does not contain rgbd data"
        return rgbd_to_video_array(x["rgbd"] for x in data)  # type: ignore[index]

    return rgbd_to_video_array(x.rgbd for x in data)  # type: ignore[union-attr]


def display_video(
    data: _ActOrStepObservationSequence,
    size: Optional[Tuple[int, int]] = None,
    fps: int = 25,
    file_format: str = "webm",
) -> None:
    if size is None:
        size = (512, 512)
    video_array = observation_video_array(data)
    visualize_arraylike_as_video(
        video_array,
        normalize=False,
        size=size,
        video_format=file_format,
        fps=fps,
    )


def visualize_worlds_in_folder(world_paths: Iterable[Path], resolution: int = 1024, num_frames: int = 20):
    episode_seed = 0
    config = create_vr_benchmark_config()

    with config.mutable_clone() as config:
        config.recording_options.resolution_x = resolution
        config.recording_options.resolution_y = resolution
    action_type = VRAction
    env = create_env(config, action_type)

    all_observations = []
    # if we want to take a few actions
    # null_action = get_null_vr_action()
    worlds_to_sort = []
    for world_path in world_paths:
        task, seed_str, difficulty_str = world_path.name.split("__")
        difficulty = float(difficulty_str.replace("_", "."))
        seed = int(seed_str)
        worlds_to_sort.append((task, difficulty, seed, world_path))

    for (task, difficulty, seed, world_path) in sorted(worlds_to_sort):
        logger.debug(f"Loading {world_path}")
        world_file = world_path / "main.tscn"
        observations = []
        observations.append(
            env.reset_nicely_with_specific_world(
                episode_seed=episode_seed,
                world_path=str(world_file),
            )
        )
        for i in range(num_frames):
            null_action = get_null_vr_action()
            obs, _ = env.act(null_action)
            observations.append(obs)

        all_observations.append(observations)

    return all_observations


DebugItemLog = TypedDict(
    "DebugItemLog",
    {
        "name": str,
        "id": str,
        # TODO not properly overridden in children
        "class": str,
        "script": str,
        "position": str,
        "velocity": str,
        "rotation": str,
    },
)


class DebugAnimalLog(DebugItemLog):
    is_frozen: bool
    is_mobile: bool
    is_alive: bool
    hit_points: int
    current_domain: str
    behavior: str
    forward_impediment: str


# TODO make serializable version with "unknown_attributes" attr to avoid fragile debug logs
class DebugLogLine(TypedDict):
    time: int
    episode: int
    items: List[Union[DebugItemLog, DebugAnimalLog]]


def _output_folder_for(env: GodotEnv, episode_seed: int) -> Path:
    dir_root = env.config.get_dir_root()
    return Path(dir_root) / f"{episode_seed:06d}"


def get_debug_json_logs(env: GodotEnv, episode_seed: Optional[int] = None) -> List[DebugLogLine]:
    assert (
        env.config.recording_options.is_debugging_output_requested
    ), "env.config.recording_options.is_debugging_output_requested is False, would not have produced a debug log"
    if episode_seed is None:
        episode_seed = env._latest_episode_seed
    is_episode_ongoing = not env.process.is_closed and episode_seed == env._latest_episode_seed
    debug_path = _output_folder_for(env, episode_seed) / "debug.json"
    return _read_debug_json_log(debug_path, is_episode_ongoing=is_episode_ongoing)


def _read_debug_json_log(debug_json_path: Path, is_episode_ongoing: bool) -> List[DebugLogLine]:
    debug_output: List[DebugLogLine] = []
    failure = None
    if is_episode_ongoing:
        wait_until_true(debug_json_path.exists, sleep_inc=0.1)
    with open(debug_json_path, "r") as line_delimited_debug_json:
        for line, frame_dict in enumerate(line_delimited_debug_json):
            try:
                debug_output.append(json.loads(frame_dict))
            except json.JSONDecodeError as e:
                failure = (line, e, frame_dict)
                break

        if failure is not None:
            line, err, frame_dict = failure
            next_line = next(line_delimited_debug_json, None)
            is_unparseable_mid_file = next_line is not None
            if is_unparseable_mid_file:
                raise ValueError(
                    f"Failed to decode debug log line {line} '{frame_dict}'."
                    f"This is likely because the episode number is being tinkered with, "
                    f"and the latest debug output is shorter than a previous one. "
                    f"Line: '{frame_dict}'"
                ) from err
            elif not is_episode_ongoing:
                raise ValueError(
                    f"Failed to decode final debug log line {line}  '{frame_dict}'. "
                    f"The epsiode may still be in progress and mid-write."
                ) from err

        return debug_output


class ActionSmoother:
    def __init__(self, decay: float = 0.999) -> None:
        self.state: dict[str, float] = {}
        self.decay = decay

    def update(self, action: dict[str, Any]):
        for k, v in action.items():
            if v.dtype == np.float32:
                if k not in self.state:
                    self.state[k] = v * 0
                else:
                    self.state[k] = v * (1 - self.decay) + self.state[k] * self.decay
            else:
                self.state[k] = v
        return self.state
