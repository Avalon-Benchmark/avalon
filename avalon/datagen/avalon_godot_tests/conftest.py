import os
from pathlib import Path
from typing import Generator

from avalon.contrib.testing_utils import fixture
from avalon.contrib.testing_utils import temp_path_
from avalon.contrib.testing_utils import use
from avalon.datagen.env_helper import create_vr_benchmark_config
from avalon.datagen.godot_env.actions import VRAction
from avalon.datagen.godot_env.goals import NullGoalEvaluator
from avalon.datagen.godot_env.godot_env import GodotEnv
from avalon.datagen.godot_env.observations import AvalonObservation
from avalon.datagen.godot_generated_types import AvalonSimSpec
from avalon.datagen.world_creation.world_generator import GeneratedAvalonWorldParams

AvalonEnv = GodotEnv[AvalonObservation, VRAction, GeneratedAvalonWorldParams]


@fixture
def frame_resolution_() -> int:
    return 256


@fixture
@use(temp_path_)
def behavior_test_folder_(temp_path: Path):
    return temp_path / "avalon_behaviors"


@fixture
@use(
    behavior_test_folder_,
    frame_resolution_,
)
def avalon_config_(
    behavior_test_folder: Path,
    frame_resolution: int,
) -> AvalonSimSpec:
    train_config: AvalonSimSpec
    with create_vr_benchmark_config().mutable_clone() as train_config:
        train_config.dir_root = os.path.join(behavior_test_folder, "runtime")
        train_config.recording_options.resolution_x = frame_resolution
        train_config.recording_options.resolution_y = frame_resolution
        train_config.recording_options.is_debugging_output_requested = True
        assert train_config.player is not None
        train_config.player.mass = 60.0
        train_config.player.is_slowed_from_crouching = False
        return train_config


@fixture
@use(avalon_config_)
def godot_env_(avalon_config: AvalonSimSpec) -> Generator[AvalonEnv, None, None]:
    env = GodotEnv(
        config=avalon_config,
        action_type=VRAction,
        observation_type=AvalonObservation,
        goal_evaluator=NullGoalEvaluator(),
        is_dev_flag_added=True,
    )
    if not "PYTEST_CURRENT_TEST" in os.environ:
        print(f"GodotEnv log: {env.process.log_path}")
    yield env
    if env._bridge.is_open:
        env.close()
