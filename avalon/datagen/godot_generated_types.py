# GENERATED FILE
# See generate_godot_code.py for details

from typing import Final
from typing import Optional

import attr

from avalon.datagen.godot_base_types import *

BRIDGE_LOG_SIGNAL: Final = "Establishing Bridge"
READY_LOG_SIGNAL: Final = "GODATA!"
FIXED_FPS: Final = 10
RGBD_FEATURE: Final = "rgbd"
ISOMETRIC_RGBD_FEATURE: Final = "isometric_rgbd"
TOP_DOWN_RGBD_FEATURE: Final = "top_down_rgbd"
EPISODE_ID_FEATURE: Final = "episode_id"
FRAME_ID_FEATURE: Final = "frame_id"
EVENT_HAPPENED_FEATURE: Final = "event_happened"
ACTION_FEATURE: Final = "action"
RESET_MESSAGE: Final = 0
SEED_MESSAGE: Final = 1
RENDER_MESSAGE: Final = 2
ACTION_MESSAGE: Final = 3
SELECT_FEATURES_MESSAGE: Final = 4
QUERY_AVAILABLE_FEATURES_MESSAGE: Final = 5
CLOSE_MESSAGE: Final = 6
DEBUG_CAMERA_ACTION_MESSAGE: Final = 7
HUMAN_INPUT_MESSAGE: Final = 9
SAVE_SNAPSHOT_MESSAGE: Final = 10
LOAD_SNAPSHOT_MESSAGE: Final = 11
VR_ACTION_SPACE: Final = "VR_ACTION_SPACE"
MOUSE_KEYBOARD_ACTION_SPACE: Final = "MOUSE_KEYBOARD_ACTION_SPACE"
LEFT_HAND: Final = "Left"
RIGHT_HAND: Final = "Right"
FAKE_TYPE_IMAGE: Final = -1
SNAPSHOT_JSON: Final = "snapshot_context.json"
SNAPSHOT_SUBPATH: Final = "snapshots"
SCENE_ROOT_NODE_PATH: Final = "/root/scene_root"
WORLD_NODE_PATH: Final = SCENE_ROOT_NODE_PATH + "/world"
DYNAMIC_TRACKER_NODE_NAME: Final = "dynamic_tracker"
DYNAMIC_TRACKER_NODE_PATH: Final = WORLD_NODE_PATH + "/Avalon/dynamic_tracker"


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class SpecBase(Serializable):
    pass


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class ControlledNodeSpec(SpecBase):
    spawn_point_name: str


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class RecordingOptionsSpec(SpecBase):
    user_id: Optional[str]
    apk_version: Optional[str]
    recorder_host: Optional[str]
    recorder_port: Optional[int]
    is_teleporter_enabled: bool
    resolution_x: int
    resolution_y: int
    is_recording_human_actions: bool
    is_remote_recording_enabled: bool
    is_adding_debugging_views: bool
    is_debugging_output_requested: bool


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class PlayerSpec(ControlledNodeSpec):
    # m/s
    max_head_linear_speed: float
    # deg/s
    max_head_angular_speed: AnyVec3
    # m/s
    max_hand_linear_speed: float
    # deg/s
    max_hand_angular_speed: float
    # m
    jump_height: float
    arm_length: float
    starting_hit_points: float
    # kg
    mass: float
    arm_mass_ratio: float
    head_mass_ratio: float
    # m/s
    standup_speed_after_climbing: float
    # m
    min_head_position_off_of_floor: float
    # N
    push_force_magnitude: float
    # N
    throw_force_magnitude: float
    starting_left_hand_position_relative_to_head: AnyVec3
    starting_right_hand_position_relative_to_head: AnyVec3
    # m/s
    minimum_fall_speed: float
    fall_damage_coefficient: float
    num_frames_alive_after_food_is_gone: int
    eat_area_radius: float
    is_displaying_debug_meshes: bool
    is_human_playback_enabled: bool
    is_slowed_from_crouching: bool


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class AgentPlayerSpec(PlayerSpec):
    pass


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class HumanPlayerSpec(PlayerSpec):
    pass


# TODO: should SimSpec still inherit from DataConfigImplementation?
@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class SimSpec(SpecBase, DataConfigImplementation):
    episode_seed: Optional[int]
    dir_root: str
    recording_options: RecordingOptionsSpec
    player: PlayerSpec


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class AvalonSimSpec(SimSpec):
    pass


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class MouseKeyboardAgentPlayerSpec(AgentPlayerSpec):
    pass


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class MouseKeyboardHumanPlayerSpec(HumanPlayerSpec):
    pass


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class VRAgentPlayerSpec(AgentPlayerSpec):
    pass


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class VRHumanPlayerSpec(HumanPlayerSpec):
    pass


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class CaregiverSimSpec(AvalonSimSpec):
    caregiver: ControlledNodeSpec
