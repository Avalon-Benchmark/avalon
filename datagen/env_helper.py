import os
from typing import List
from typing import Optional
from typing import Type

import attr
import torch
from einops import rearrange

from common.errors import SwitchError
from common.utils import DATA_FOLDER
from common.visual_utils import visualize_tensor_as_video
from datagen.godot_base_types import Vector3
from datagen.godot_env import AttrsAction
from datagen.godot_env import AvalonObservationType
from datagen.godot_env import GoalEvaluator
from datagen.godot_env import GoalProgressResult
from datagen.godot_env import GodotEnv
from datagen.godot_env import MouseKeyboardActionType
from datagen.godot_env import VRActionType
from datagen.godot_generated_types import AgentPlayerSpec
from datagen.godot_generated_types import AvalonSimSpec
from datagen.godot_generated_types import MouseKeyboardAgentPlayerSpec
from datagen.godot_generated_types import MouseKeyboardHumanPlayerSpec
from datagen.godot_generated_types import RecordingOptionsSpec
from datagen.godot_generated_types import VRAgentPlayerSpec
from datagen.godot_generated_types import VRHumanPlayerSpec
from datagen.world_creation.world_generator import GenerateWorldParams


def create_base_benchmark_config() -> AvalonSimSpec:
    return AvalonSimSpec(
        dataset_id=0,
        label=0,
        video_min=0,
        video_max=750,
        frame_max=100,
        random_int=0,
        random_key="{dir_root}",
        dir_root=os.path.join(DATA_FOLDER, "interactive"),
        output_file_name_format="{dir_root}/{video_id}/{feature}__{type_and_dim}.raw",
        scene_path="res://scenes/empty.tscn",
        is_using_shared_caches=False,
        is_generating_paired_videos=False,
        player=AgentPlayerSpec(
            max_head_linear_speed=0.3,
            max_head_angular_speed=Vector3(5.0, 10.0, 1.0),
            max_hand_linear_speed=1.0,
            max_hand_angular_speed=15.0,
            jump_height=1.5,
            arm_length=1.5,
            starting_hit_points=100.0,
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
            fall_damage_coefficient=0.0,
            total_energy_coefficient=0.0,
            body_kinetic_energy_coefficient=0.0,
            body_potential_energy_coefficient=0.0,
            head_potential_energy_coefficient=0.0,
            left_hand_kinetic_energy_coefficient=0.0,
            left_hand_potential_energy_coefficient=0.0,
            right_hand_kinetic_energy_coefficient=0.0,
            right_hand_potential_energy_coefficient=0.0,
            num_frames_alive_after_food_is_gone=50,
            eat_area_radius=0.5,
            is_displaying_debug_meshes=False,
        ),
        recording_options=RecordingOptionsSpec(
            user_id=None,
            apk_version=None,
            recorder_host=None,
            recorder_port=None,
            is_teleporter_enabled=False,
            resolution_x=64,
            resolution_y=64,
            is_recording_rgb=True,
            is_recording_depth=False,
            is_recording_human_actions=False,
            is_remote_recording_enabled=False,
            is_adding_debugging_views=False,
            is_debugging_output_requested=False,
        ),
    )


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
    return MouseKeyboardActionType(
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


def get_null_vr_action() -> AttrsAction:
    return VRActionType(
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


def create_env(config: AvalonSimSpec, action_type: Type[AttrsAction]) -> GodotEnv:
    return GodotEnv(
        config=config,
        observation_type=AvalonObservationType,
        action_type=action_type,
        goal_evaluator=NullGoalEvaluator(),
        gpu_id=0,
    )


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class NullGoalEvaluator(GoalEvaluator[AvalonObservationType]):
    def calculate_goal_progress(self, observation: AvalonObservationType) -> GoalProgressResult:
        return GoalProgressResult(reward=0, is_done=False, log={})

    def reset(self, observation: AvalonObservationType, world_params: Optional[GenerateWorldParams] = None) -> None:
        pass


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class PlaybackGoalEvaluator(GoalEvaluator[AvalonObservationType]):
    def calculate_goal_progress(self, observation: AvalonObservationType) -> GoalProgressResult:
        return GoalProgressResult(reward=0, is_done=False, log={})

    def reset(self, observation: AvalonObservationType, world_params: Optional[GenerateWorldParams] = None) -> None:
        pass


def display_video(data: List[AvalonObservationType], size=None):
    if size is None:
        size = (512, 512)
    tensor = torch.stack(
        [
            rearrange(
                torch.flipud(torch.tensor(x.rgbd[:, :, :3])),
                "h w c -> c h w",
            )
            / 255.0
            for x in data
        ]
    )
    visualize_tensor_as_video(tensor, normalize=False, size=size)


def get_action_type_from_config(config: AvalonSimSpec) -> Type[AttrsAction]:
    if isinstance(config.player, MouseKeyboardAgentPlayerSpec) or isinstance(
        config.player, MouseKeyboardHumanPlayerSpec
    ):
        return MouseKeyboardActionType
    elif isinstance(config.player, VRAgentPlayerSpec) or isinstance(config.player, VRHumanPlayerSpec):
        return VRActionType
    else:
        raise SwitchError(config.player)
