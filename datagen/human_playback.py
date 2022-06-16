import json
import os
from collections import OrderedDict
from io import BufferedReader
from math import prod
from pathlib import Path
from typing import List
from typing import Optional
from typing import OrderedDict as OrderedDictType
from typing import Tuple
from typing import Type
from typing import cast

import attr
import numpy as np
import numpy.typing as npt

from common.errors import SwitchError
from datagen.godot_env import FAKE_TYPE_IMAGE
from datagen.godot_env import NP_DTYPE_MAP
from datagen.godot_env import AttrsAction
from datagen.godot_env import AvalonObservationType
from datagen.godot_env import GodotEnvActionLog
from datagen.godot_env import _from_bytes
from datagen.godot_generated_types import CLOSE_MESSAGE
from datagen.godot_generated_types import HUMAN_INPUT_MESSAGE
from datagen.godot_generated_types import QUERY_AVAILABLE_FEATURES_MESSAGE
from datagen.godot_generated_types import RESET_MESSAGE
from datagen.godot_generated_types import SEED_MESSAGE
from datagen.godot_generated_types import SELECT_FEATURES_MESSAGE
from datagen.godot_generated_types import AvalonSimSpec
from datagen.godot_generated_types import MouseKeyboardAgentPlayerSpec
from datagen.godot_generated_types import MouseKeyboardHumanPlayerSpec
from datagen.godot_generated_types import RecordingOptionsSpec
from datagen.godot_generated_types import VRAgentPlayerSpec
from datagen.godot_generated_types import VRHumanPlayerSpec


def _write_message(fp, message_type: int, message: bytes):
    fp.write(message_type.to_bytes(1, byteorder="little", signed=False))
    fp.write(message)


def get_replay_log_from_human_recording(
    new_action_record_path: str,
    action_record_path: str,
    metadata_path: str,
    worlds_path: str,
    selected_features,
    action_type: Type[AttrsAction],
) -> GodotEnvActionLog:
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    seed = metadata["seed"]
    video_id = metadata["video_id"]
    world_id = metadata["world_id"]
    world_path = get_world_path_from_world_id(root_path=Path(worlds_path), world_id=world_id)

    with open(action_record_path, "rb") as f:
        content = f.read()

    Path(new_action_record_path).unlink(missing_ok=True)

    # need to manually add some header messages to our replay log
    with open(new_action_record_path, "wb") as f:
        _write_message(f, QUERY_AVAILABLE_FEATURES_MESSAGE, bytes())

        size_doubleword = (len(selected_features)).to_bytes(4, byteorder="little", signed=False)
        feature_names_bytes = ("\n".join(selected_features.keys()) + "\n").encode("UTF-8")
        _write_message(f, SELECT_FEATURES_MESSAGE, size_doubleword + feature_names_bytes)

        seed_message_bytes = seed.to_bytes(8, byteorder="little", signed=True)
        seed_message_bytes += video_id.to_bytes(8, byteorder="little", signed=True)
        _write_message(f, SEED_MESSAGE, seed_message_bytes)

        reset_message_bytes = (world_path + "\n").encode("UTF-8") + action_type.get_null_action().to_bytes()
        _write_message(f, RESET_MESSAGE, reset_message_bytes)

        f.write(content)

        _write_message(f, CLOSE_MESSAGE, bytes())

    return GodotEnvActionLog.parse(new_action_record_path, action_type)


def get_observations_from_human_recording(
    observations_path: str, available_features: OrderedDictType[str, Tuple[int, Tuple[int, ...]]]
):
    file_size = os.stat(observations_path).st_size
    observations = []
    with open(observations_path, "rb") as record_log:
        while record_log.tell() < file_size:
            feature_data = {}
            for feature_name, (data_type, dims) in available_features.items():
                if feature_name in {"depth", "rgb"}:
                    continue
                size = prod(dims)
                if data_type != FAKE_TYPE_IMAGE:
                    size = size * 4
                feature_data[feature_name] = _read_shape(record_log, dims, size, dtype=NP_DTYPE_MAP[data_type])

            cleaned_feature_data = {
                feature.name: feature_data.get(feature.name, np.array([]))
                for feature in attr.fields(AvalonObservationType)
            }
            observations.append(AvalonObservationType(**cleaned_feature_data))
    return observations


def _read_shape(
    record_log: BufferedReader,
    shape: Tuple[int, ...],
    size: Optional[int] = None,
    dtype: npt.DTypeLike = np.uint8,
) -> npt.NDArray:
    byte_buffer = record_log.read(size if size is not None else prod(shape))
    return np.ndarray(shape=shape, dtype=dtype, buffer=byte_buffer)


def get_world_path_from_world_id(root_path: Path, world_id: str) -> str:
    return str(root_path / world_id / "main.tscn")


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class MouseKeyboardHumanInputType(AttrsAction):
    is_move_forward_pressed: float
    is_move_backward_pressed: float
    is_move_left_pressed: float
    is_move_right_pressed: float
    is_jump_pressed: float
    is_eat_pressed: float
    is_grab_pressed: float
    is_throw_pressed: float
    is_crouch_pressed: float
    is_wheel_up_just_released: float
    is_wheel_down_just_released: float
    is_mouse_mode_toggled: float
    is_active_hand_toggled: float


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class VRHumanInputType(AttrsAction):
    arvr_origin_position_x: float
    arvr_origin_position_y: float
    arvr_origin_position_z: float
    arvr_origin_rotation_x: float
    arvr_origin_rotation_y: float
    arvr_origin_rotation_z: float
    arvr_camera_position_x: float
    arvr_camera_position_y: float
    arvr_camera_position_z: float
    arvr_camera_rotation_x: float
    arvr_camera_rotation_y: float
    arvr_camera_rotation_z: float
    arvr_left_hand_position_x: float
    arvr_left_hand_position_y: float
    arvr_left_hand_position_z: float
    arvr_left_hand_rotation_x: float
    arvr_left_hand_rotation_y: float
    arvr_left_hand_rotation_z: float
    arvr_right_hand_position_x: float
    arvr_right_hand_position_y: float
    arvr_right_hand_position_z: float
    arvr_right_hand_rotation_x: float
    arvr_right_hand_rotation_y: float
    arvr_right_hand_rotation_z: float
    look_stick_x: float
    look_stick_y: float
    strafe_stick_x: float
    strafe_stick_y: float
    is_left_hand_grab_pressed: float
    is_right_hand_grab_pressed: float
    is_jump_pressed: float


def parse_human_input(path: str, human_input_type: Type[AttrsAction]) -> List[AttrsAction]:
    human_inputs = []
    with open(path, "rb") as f:
        while message_bytes := f.read(1):
            message = int.from_bytes(message_bytes, byteorder="little", signed=False)
            assert message == HUMAN_INPUT_MESSAGE
            size_bytes = f.read(4)
            size, _ = _from_bytes(int, size_bytes)
            human_input_bytes = size_bytes + f.read(cast(int, size))
            human_inputs.append(human_input_type.from_bytes(human_input_bytes))

    return human_inputs


def get_oculus_config_for_human_playback() -> AvalonSimSpec:
    # NOTE: make sure you get the type of player correct in config.json
    config = AvalonSimSpec.from_dict(json.load(open("datagen/godot/android/config.json", "r")))

    # update player to be agent player
    with config.mutable_clone() as config:
        # update recording options to make sense for the agent
        config.recording_options = RecordingOptionsSpec(
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
        )
        if isinstance(config.player, MouseKeyboardHumanPlayerSpec):
            config.player = MouseKeyboardAgentPlayerSpec.from_dict(config.player.to_dict())
        elif isinstance(config.player, VRHumanPlayerSpec):
            player_dict = config.player.to_dict()
            player_dict["arm_length"] = 10_000
            player_dict["total_energy_coefficient"] = 1e-4
            # make vr arm length basically infinity
            config.player = VRAgentPlayerSpec.from_dict(player_dict)
        else:
            raise SwitchError(config.player)

    return config


DEFAULT_AVAILABLE_FEATURES = OrderedDict(
    [
        ("physical_body_position", (7, (3,))),
        ("physical_head_position", (7, (3,))),
        ("physical_left_hand_position", (7, (3,))),
        ("physical_right_hand_position", (7, (3,))),
        ("physical_body_rotation", (7, (3,))),
        ("physical_head_rotation", (7, (3,))),
        ("physical_left_hand_rotation", (7, (3,))),
        ("physical_right_hand_rotation", (7, (3,))),
        ("physical_body_linear_velocity", (7, (3,))),
        ("physical_head_linear_velocity", (7, (3,))),
        ("physical_left_hand_linear_velocity", (7, (3,))),
        ("physical_right_hand_linear_velocity", (7, (3,))),
        ("physical_head_angular_velocity", (7, (3,))),
        ("physical_left_hand_angular_velocity", (7, (3,))),
        ("physical_right_hand_angular_velocity", (7, (3,))),
        ("physical_body_delta_position", (7, (3,))),
        ("physical_head_delta_position", (7, (3,))),
        ("physical_left_hand_delta_position", (7, (3,))),
        ("physical_right_hand_delta_position", (7, (3,))),
        ("physical_head_relative_position", (7, (3,))),
        ("physical_left_hand_relative_position", (7, (3,))),
        ("physical_right_hand_relative_position", (7, (3,))),
        ("physical_head_relative_rotation", (7, (3,))),
        ("physical_left_hand_relative_rotation", (7, (3,))),
        ("physical_right_hand_relative_rotation", (7, (3,))),
        ("physical_body_delta_rotation", (7, (3,))),
        ("physical_head_delta_rotation", (7, (3,))),
        ("physical_left_hand_delta_rotation", (7, (3,))),
        ("physical_right_hand_delta_rotation", (7, (3,))),
        ("physical_body_delta_linear_velocity", (7, (3,))),
        ("physical_head_delta_linear_velocity", (7, (3,))),
        ("physical_left_hand_delta_linear_velocity", (7, (3,))),
        ("physical_right_hand_delta_linear_velocity", (7, (3,))),
        ("physical_head_delta_angular_velocity", (7, (3,))),
        ("physical_left_hand_delta_angular_velocity", (7, (3,))),
        ("physical_right_hand_delta_angular_velocity", (7, (3,))),
        ("left_hand_thing_colliding_with_hand", (2, (1,))),
        ("left_hand_held_thing", (2, (1,))),
        ("right_hand_thing_colliding_with_hand", (2, (1,))),
        ("right_hand_held_thing", (2, (1,))),
        ("physical_body_kinetic_energy_expenditure", (3, (1,))),
        ("physical_body_potential_energy_expenditure", (3, (1,))),
        ("physical_head_potential_energy_expenditure", (3, (1,))),
        ("physical_left_hand_kinetic_energy_expenditure", (3, (1,))),
        ("physical_left_hand_potential_energy_expenditure", (3, (1,))),
        ("physical_right_hand_kinetic_energy_expenditure", (3, (1,))),
        ("physical_right_hand_potential_energy_expenditure", (3, (1,))),
        ("total_energy_expenditure", (3, (1,))),
        ("fall_damage", (3, (1,))),
        ("hit_points_lost_from_enemies", (3, (1,))),
        ("hit_points_gained_from_eating", (3, (1,))),
        ("reward", (3, (1,))),
        ("hit_points", (3, (1,))),
        ("is_dead", (2, (1,))),
        ("nearest_food_position", (7, (3,))),
        ("nearest_food_id", (2, (1,))),
        ("is_food_present_in_world", (2, (1,))),
        ("is_done", (2, (1,))),
        ("video_id", (2, (1,))),
        ("frame_id", (2, (1,))),
        ("rgb", (-1, (64, 64, 3))),
        ("depth", (-1, (64, 64, 3))),
    ]
)


DEFAULT_SELECTED_FEATURES = OrderedDict(
    [
        ("physical_body_position", (7, (3,))),
        ("physical_head_position", (7, (3,))),
        ("physical_left_hand_position", (7, (3,))),
        ("physical_right_hand_position", (7, (3,))),
        ("physical_body_rotation", (7, (3,))),
        ("physical_head_rotation", (7, (3,))),
        ("physical_left_hand_rotation", (7, (3,))),
        ("physical_right_hand_rotation", (7, (3,))),
        ("physical_body_delta_position", (7, (3,))),
        ("physical_head_delta_position", (7, (3,))),
        ("physical_left_hand_delta_position", (7, (3,))),
        ("physical_right_hand_delta_position", (7, (3,))),
        ("physical_body_delta_rotation", (7, (3,))),
        ("physical_head_delta_rotation", (7, (3,))),
        ("physical_left_hand_delta_rotation", (7, (3,))),
        ("physical_right_hand_delta_rotation", (7, (3,))),
        ("left_hand_thing_colliding_with_hand", (2, (1,))),
        ("left_hand_held_thing", (2, (1,))),
        ("right_hand_thing_colliding_with_hand", (2, (1,))),
        ("right_hand_held_thing", (2, (1,))),
        ("physical_body_kinetic_energy_expenditure", (3, (1,))),
        ("physical_body_potential_energy_expenditure", (3, (1,))),
        ("physical_head_potential_energy_expenditure", (3, (1,))),
        ("physical_left_hand_kinetic_energy_expenditure", (3, (1,))),
        ("physical_left_hand_potential_energy_expenditure", (3, (1,))),
        ("physical_right_hand_kinetic_energy_expenditure", (3, (1,))),
        ("physical_right_hand_potential_energy_expenditure", (3, (1,))),
        ("total_energy_expenditure", (3, (1,))),
        ("fall_damage", (3, (1,))),
        ("hit_points_lost_from_enemies", (3, (1,))),
        ("hit_points_gained_from_eating", (3, (1,))),
        ("hit_points", (3, (1,))),
        ("is_dead", (2, (1,))),
        ("reward", (3, (1,))),
        ("video_id", (2, (1,))),
        ("frame_id", (2, (1,))),
        ("rgb", (-1, (64, 64, 3))),
    ]
)

# TODO
# def rollout_human_playback(replay_log: GodotEnvActionLog) -> List[AvalonObservationType]:
#     env = create_env(config, action_type)
#     seed, video_id = replay_log.initial_seed, replay_log.initial_video_id
#     world_id = metadata["world_id"]
#     world_path = get_world_path_from_world_id(root_path=Path(worlds_path), world_id=world_id)
#     env.seed_nicely(seed, video_id)
#
#     actions = []
#     observations = []
#     for message in replay_log.messages:
#         if message[0] == ACTION_MESSAGE:
#             action = message[1]
#             obs, _ = env.act(cast(action_type, action))
#             actions.append(action)
#             observations.append(obs)
#         elif message[0] == RESET_MESSAGE:
#             observations.append(
#                 env.reset_nicely_with_specific_world(seed=seed, world_id=video_id, world_path=world_path)
#             )
#         else:
#             raise SwitchError(f"Invalid replay message {message}")
#
#     return observations
