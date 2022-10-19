import struct
from typing import Any
from typing import Dict
from typing import Protocol
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from typing import cast

import attr
import gym
import numpy as np
from gym import spaces
from gym.spaces import Box

from avalon.common.errors import SwitchError
from avalon.datagen.godot_base_types import Vector3


class ActionProtocol(Protocol):
    @classmethod
    def to_gym_space(cls) -> spaces.Space:
        ...

    @classmethod
    def from_input(cls: Type["ActionType"], input_vec: Dict[str, np.ndarray]) -> "ActionType":
        ...

    def to_bytes(self) -> bytes:
        """Convert this action to bytes to be sent to godot"""

    @classmethod
    def from_bytes(cls: Type["ActionType"], action_bytes: bytes) -> "ActionType":
        """Parse this action from bytes from an action log"""

    @classmethod
    def get_null_action(cls: Type["ActionType"]) -> "ActionType":
        """An empty action that is sent with reset messages"""


ActionType = TypeVar("ActionType", bound=ActionProtocol)


class AttrsAction(ActionProtocol):
    @classmethod
    def to_gym_space(cls) -> spaces.Space:
        """Basic float space the same dimensions as this action.

        Override if any customization is necessary (i.e. discrete spaces)
        """
        field_counts = len(attr.fields(cls))
        return spaces.Box(
            low=-np.ones((field_counts,), dtype=np.float32) / 10,
            high=np.ones((field_counts,), dtype=np.float32) / 10,
            shape=(field_counts,),
            dtype=np.float32,
        )

    @classmethod
    def from_input(cls: Type["ActionType"], input_vec: Dict[str, np.ndarray]) -> "ActionType":
        raise NotImplementedError

    @classmethod
    def get_null_action(cls: Type["ActionType"]) -> "ActionType":
        action_fields = [x for x in attr.fields(cls)]
        null_action_kwargs = {}
        for field in action_fields:
            if field.type == int:
                value: Union[int, float] = 0
            elif field.type == float:
                value = 0.0
            else:
                raise SwitchError(field.type)
            null_action_kwargs[field.name] = value

        return cls(**null_action_kwargs)

    def to_bytes(self) -> bytes:
        action_bytes = b"".join(
            _to_bytes(
                cast(Type, field.type),
                getattr(self, field.name),
            )
            for field in attr.fields(self.__class__)
        )
        return _to_bytes(int, len(action_bytes)) + action_bytes

    @classmethod
    def from_bytes(cls: Type["ActionType"], action_bytes: bytes) -> "ActionType":
        fields: Dict[str, Any] = {}
        _size, remaining_bytes = _from_bytes(int, action_bytes)
        for field in attr.fields(cls):
            fields[field.name], remaining_bytes = _from_bytes(cast(Type, field.type), remaining_bytes)
        return cls(**fields)


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class VRAction(AttrsAction):
    """All of these that are floats have range (-1, 1). They are rescaled to physical units in the simulator."""

    head_x: float
    head_y: float
    head_z: float
    head_pitch: float
    head_yaw: float
    head_roll: float
    left_hand_x: float
    left_hand_y: float
    left_hand_z: float
    left_hand_pitch: float
    left_hand_yaw: float
    left_hand_roll: float
    right_hand_x: float
    right_hand_y: float
    right_hand_z: float
    right_hand_pitch: float
    right_hand_yaw: float
    right_hand_roll: float
    is_left_hand_grasping: float
    is_right_hand_grasping: float
    is_jumping: float

    @classmethod
    def to_gym_space(cls) -> spaces.Space:
        return spaces.Dict(
            {
                "real": Box(low=-1, high=1, shape=(18,)),
                "discrete": gym.spaces.MultiBinary(3),
            }
        )

    @classmethod
    def from_input(cls, input_dict: Dict[str, np.ndarray]) -> "VRAction":
        # clipping each triplet to sphere
        input_real = input_dict["real"]
        triplet_norm = np.linalg.norm(np.reshape(input_real, (6, 3)), axis=-1)
        scale = 1 / np.clip(triplet_norm, a_min=1, a_max=float("inf"))
        clipped_real = np.repeat(scale, 3) * input_real
        input_vec = np.concatenate([clipped_real, input_dict["discrete"]], axis=-1)
        action = cls(*tuple(x.item() for x in input_vec))
        return action


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class MouseKeyboardAction(AttrsAction):
    head_x: float
    head_z: float
    head_pitch: float
    head_yaw: float
    is_left_hand_grasping: float
    is_right_hand_grasping: float
    is_left_hand_throwing: float
    is_right_hand_throwing: float
    is_jumping: float
    is_eating: float
    is_crouching: float

    @classmethod
    def to_gym_space(cls) -> spaces.Space:
        # TODO: move this back to 2 spaces for perf gains
        return spaces.Dict(
            {
                "real": Box(low=-1, high=1, shape=(4,)),
                "discrete": gym.spaces.MultiBinary(7),
            }
        )

    @classmethod
    def from_input(cls, input_dict: Dict[str, np.ndarray]) -> "MouseKeyboardAction":
        clipped_real = np.clip(input_dict["real"], a_min=-1, a_max=1)
        input_vec = np.concatenate([clipped_real, input_dict["discrete"]], axis=-1)

        return cls(*tuple(x.item() for x in input_vec))


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True, slots=True)
class DebugCameraAction(AttrsAction):
    offset: Vector3 = Vector3(0, 0, 0)
    rotation: Vector3 = Vector3(0, 0, 0)
    is_facing_tracked: bool = False
    # Setting to False will result in an empty response for the first frame,
    # but will not interfere with physics or the results of surounding actions (i.e. when replaying)
    is_frame_advanced: bool = True
    tracking_node: str = "physical_head"

    @classmethod
    def isometric(
        cls,
        tracking_node: str = "physical_head",
        distance: float = 12,
        is_facing_tracked: bool = True,
        is_frame_advanced: bool = True,
    ):
        return DebugCameraAction(
            Vector3(distance / 2, abs(distance), distance / 2),
            Vector3(np.radians(-45), np.radians(45), 0),
            is_facing_tracked,
            is_frame_advanced,
            tracking_node,
        )

    @classmethod
    def level(
        cls,
        tracking_node: str = "physical_head",
        distance: float = 12,
        is_facing_tracked: bool = True,
        is_frame_advanced: bool = True,
    ):
        return DebugCameraAction(
            Vector3(distance / 2, 0, distance / 2),
            Vector3(np.radians(-45), 0, 0),
            is_facing_tracked,
            is_frame_advanced,
            tracking_node,
        )

    def to_bytes(self) -> bytes:
        action_bytes = b""
        for vec in [self.offset, self.rotation]:
            action_bytes += _to_bytes(float, vec.x)
            action_bytes += _to_bytes(float, vec.y)
            action_bytes += _to_bytes(float, vec.z)
        action_bytes += _to_bytes(float, 1.0 if self.is_facing_tracked else 0.0)
        action_bytes += _to_bytes(float, 1.0 if self.is_frame_advanced else 0.0)
        action_bytes += self.tracking_node.encode("UTF-8")
        return _to_bytes(int, len(action_bytes)) + action_bytes

    @classmethod
    def _read_vec(cls, action_bytes: bytes) -> Tuple[Vector3, bytes]:
        v = [0.0, 1.0, 2.0]
        remaining = action_bytes
        for axis in v:
            v[axis], remaining = _from_bytes(float, action_bytes)  # type: ignore[call-overload]
        return Vector3(*v), remaining

    @classmethod
    def from_bytes(cls, action_bytes: bytes) -> "DebugCameraAction":
        _size, remaining_bytes = _from_bytes(int, action_bytes)
        offset, remaining_bytes = cls._read_vec(remaining_bytes)
        rotation, remaining_bytes = cls._read_vec(remaining_bytes)
        is_facing_tracked, remaining_bytes = _from_bytes(float, remaining_bytes)
        is_frame_advanced, remaining_bytes = _from_bytes(float, remaining_bytes)
        tracking_node = remaining_bytes.decode("UTF-8")
        return cls(offset, rotation, is_facing_tracked != 0, is_frame_advanced != 0, tracking_node)


def _to_bytes(value_type: Type, value: Any) -> bytes:
    if value_type is float:
        return struct.pack("<f", value)
    if value_type is int:
        return struct.pack("<i", value)
    raise SwitchError(f"{value_type} (value={value})")


def _from_bytes(value_type: Type, value_bytes: bytes) -> Tuple[Union[int, float], bytes]:
    byte_format = ""
    if value_type is float:
        byte_format = "<f"
    elif value_type is int:
        byte_format = "<i"
    else:
        raise SwitchError(f"{value_type} should be either float or int")

    size = struct.calcsize(byte_format)
    bytes_to_consume, bytes_remaining = value_bytes[:size], value_bytes[size:]
    return struct.unpack(byte_format, bytes_to_consume)[0], bytes_remaining
