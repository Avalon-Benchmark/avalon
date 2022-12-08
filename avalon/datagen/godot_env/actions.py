import struct
from typing import Any
from typing import ClassVar
from typing import Dict
from typing import Generic
from typing import Protocol
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from typing import cast
from typing import runtime_checkable

import attr
import gym
import numpy as np
from gym import spaces
from gym.spaces import Box

from avalon.common.errors import SwitchError
from avalon.datagen.godot_base_types import Vector3


@runtime_checkable
class ActionProtocol(Protocol):
    @classmethod
    def to_gym_space(cls) -> spaces.Space:
        ...

    @classmethod
    def from_input(cls: Type["ActionType"], input_dict: Dict[str, np.ndarray]) -> "ActionType":
        ...

    def to_bytes(self) -> bytes:
        """Convert this action to bytes to be sent to godot"""

    @classmethod
    def _from_bytes_with_remainder(cls: Type["ActionType"], action_bytes: bytes) -> Tuple["ActionType", bytes]:
        """Parse this action from bytes from an action log, including unused bytes in the return"""

    @classmethod
    def from_bytes(cls: Type["ActionType"], action_bytes: bytes) -> "ActionType":
        """Parse this action from bytes from an action log"""
        return cls._from_bytes_with_remainder(action_bytes)[0]

    @classmethod
    def get_null_action(cls: Type["ActionType"]) -> "ActionType":
        """An empty action that is sent with reset messages"""


ActionType = TypeVar("ActionType", bound=ActionProtocol)
ActionType2 = TypeVar("ActionType2", bound=ActionProtocol)


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
    def from_input(cls: Type[ActionType], input_dict: Dict[str, np.ndarray]) -> ActionType:
        raise NotImplementedError

    @classmethod
    def get_null_action(cls: Type[ActionType]) -> ActionType:
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
    def _from_bytes_with_remainder(cls: Type[ActionType], action_bytes: bytes) -> Tuple[ActionType, bytes]:
        fields: Dict[str, Any] = {}
        _size, remaining_bytes = _from_bytes(int, action_bytes)
        for field in attr.fields(cls):
            fields[field.name], remaining_bytes = _from_bytes(cast(Type, field.type), remaining_bytes)
        return cls(**fields), remaining_bytes


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


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class CombinedAction(ActionProtocol, Generic[ActionType, ActionType2]):

    actions: Tuple[ActionType, ActionType2]

    SPACE_DICT_PREFIX: ClassVar = "combined_action"

    @classmethod
    def get_action_types(
        cls: "Type[CombinedAction[ActionType, ActionType2]]",
    ) -> Tuple[Type[ActionType], Type[ActionType2]]:
        "Get the action types, in order, to combine"
        raise ValueError(f"Inheritors must implement get_action_types themselves")

    @classmethod
    def _get_action_types_downcast_to_satisfy_mypy(
        cls: "Type[CombinedAction[ActionType, ActionType2]]",
    ) -> Tuple[Type[ActionProtocol], Type[ActionProtocol]]:
        # mypy can't tell tuple iterations should be unions :(
        a, b = cls.get_action_types()
        assert issubclass(a, ActionProtocol)
        assert issubclass(b, ActionProtocol)
        return (a, b)

    @classmethod
    def _prefixed_space_key(cls, key: str, index: int) -> str:
        return f"{cls.SPACE_DICT_PREFIX}_{index}_{str}"

    @classmethod
    def to_gym_space(cls: "Type[CombinedAction[ActionType, ActionType2]]") -> spaces.Space:
        action_types = cls._get_action_types_downcast_to_satisfy_mypy()
        space = {}
        for i, action_type in enumerate(action_types):
            action_space = action_type.to_gym_space()
            assert isinstance(
                action_space, spaces.Dict
            ), "to_gym_space support for CombinedActions not represented as dicts is not yet supported"
            space.update({cls._prefixed_space_key(key, i): value for key, value in action_space.items()})

        return spaces.Dict(space)

    @classmethod
    def from_input(
        cls: "Type[CombinedAction[ActionType, ActionType2]]", input_dict: Dict[str, np.ndarray]
    ) -> "CombinedAction[ActionType, ActionType2]":
        action_types = cls._get_action_types_downcast_to_satisfy_mypy()
        actions = []
        for i, a_type in enumerate(action_types):
            prefix = f"{cls.SPACE_DICT_PREFIX}_{i}_"
            i_input_dict = {k[2:]: v for k, v in input_dict.items() if k.startswith(prefix)}
            actions.append(a_type.from_input(i_input_dict))
        return cls(cast(Tuple[ActionType, ActionType2], tuple(actions)))

    @classmethod
    def get_null_action(
        cls: "Type[CombinedAction[ActionType, ActionType2]]",
    ) -> "CombinedAction[ActionType, ActionType2]":
        action_types = cls._get_action_types_downcast_to_satisfy_mypy()
        null_actions = tuple(a_type.get_null_action() for a_type in action_types)
        return cls(cast(Tuple[ActionType, ActionType2], null_actions))

    def _just_field_bytes(self, action: ActionProtocol) -> bytes:
        size, field_bytes = _from_bytes(int, action.to_bytes())
        return field_bytes

    def to_bytes(self) -> bytes:
        # mypy can't tell tuple iterations should be unions
        actions = (cast(ActionProtocol, a) for a in self.actions)
        all_action_bytes = b"".join((self._just_field_bytes(a) for a in actions))
        return _to_bytes(int, len(all_action_bytes)) + all_action_bytes

    @classmethod
    def _from_bytes_with_remainder(
        cls: "Type[CombinedAction[ActionType, ActionType2]]", action_bytes: bytes
    ) -> Tuple["CombinedAction[ActionType, ActionType2]", bytes]:
        action_types = cls._get_action_types_downcast_to_satisfy_mypy()
        remaining_bytes = action_bytes
        actions = []
        for a_type in action_types:
            action, remaining_bytes = a_type._from_bytes_with_remainder(remaining_bytes)
            actions.append(action)
        final_actions = cast(Tuple[ActionType, ActionType2], tuple(actions))
        return cls(final_actions), remaining_bytes


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
