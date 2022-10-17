from io import BufferedReader
from typing import Callable
from typing import Generic
from typing import Iterator
from typing import List
from typing import Literal
from typing import Set
from typing import Tuple
from typing import Type
from typing import Union
from typing import cast

import attr
from loguru import logger

from avalon.common.errors import SwitchError
from avalon.datagen.godot_env.actions import ActionType
from avalon.datagen.godot_env.actions import DebugCameraAction
from avalon.datagen.godot_env.actions import _from_bytes
from avalon.datagen.godot_generated_types import ACTION_MESSAGE
from avalon.datagen.godot_generated_types import CLOSE_MESSAGE
from avalon.datagen.godot_generated_types import DEBUG_CAMERA_ACTION_MESSAGE
from avalon.datagen.godot_generated_types import HUMAN_INPUT_MESSAGE
from avalon.datagen.godot_generated_types import LOAD_SNAPSHOT_MESSAGE
from avalon.datagen.godot_generated_types import QUERY_AVAILABLE_FEATURES_MESSAGE
from avalon.datagen.godot_generated_types import RENDER_MESSAGE
from avalon.datagen.godot_generated_types import RESET_MESSAGE
from avalon.datagen.godot_generated_types import SAVE_SNAPSHOT_MESSAGE
from avalon.datagen.godot_generated_types import SEED_MESSAGE
from avalon.datagen.godot_generated_types import SELECT_FEATURES_MESSAGE

_BINARY_READ: Literal["br"] = "br"

_NoPayloadMessageTypes = Literal[2, 5, 6, 10]
_NoPayloadMessage = Tuple[_NoPayloadMessageTypes]
_SeedMessage = Tuple[Literal[1], int]
_SelectFeaturesMessage = Tuple[Literal[4], List[str]]
_LoadSnapshotMessage = Tuple[Literal[11], str]
_ActionMessage = Tuple[Literal[3, 9], ActionType]
_DebugMessage = Tuple[Literal[7], DebugCameraAction]
_ResetMessage = Tuple[Literal[3, 0], ActionType, int, str, float]
_RawMessage = Union[
    _NoPayloadMessage,
    _SeedMessage,
    _SelectFeaturesMessage,
    _ActionMessage[ActionType],
    _ResetMessage[ActionType],
    _DebugMessage,
    _LoadSnapshotMessage,
]
_no_payload_messages: Set[_NoPayloadMessageTypes] = {
    RENDER_MESSAGE,
    QUERY_AVAILABLE_FEATURES_MESSAGE,
    CLOSE_MESSAGE,
    SAVE_SNAPSHOT_MESSAGE,
}

_ActionOrDebugMessage = Union[_ActionMessage[ActionType], _DebugMessage]


def _parse_raw_message_log(
    record_log: BufferedReader,
    action_from_bytes: Callable[[bytes], ActionType],
) -> Iterator[_RawMessage[ActionType]]:
    while message_bytes := record_log.read(1):
        message = int.from_bytes(message_bytes, byteorder="little", signed=False)
        if message in _no_payload_messages:
            # TODO remove cast when mypy can refine types (https://github.com/python/mypy/issues/12535)
            yield cast(_NoPayloadMessage, (message,))

        elif message == SEED_MESSAGE:
            episode_seed = int.from_bytes(record_log.read(8), byteorder="little", signed=True)
            yield cast(_SeedMessage, (message, episode_seed))

        elif message == SELECT_FEATURES_MESSAGE:
            count = int.from_bytes(record_log.read(4), byteorder="little", signed=False)
            feature_names = list(record_log.readline().decode("UTF-8")[:-1] for _ in range(count))
            yield cast(_SelectFeaturesMessage, (message, feature_names))

        elif message in (
            ACTION_MESSAGE,
            HUMAN_INPUT_MESSAGE,
            DEBUG_CAMERA_ACTION_MESSAGE,
        ):
            size_bytes = record_log.read(4)
            size, _ = _from_bytes(int, size_bytes)
            action_bytes = size_bytes + record_log.read(cast(int, size))
            yield cast(_ActionOrDebugMessage, (message, action_from_bytes(action_bytes)))

        elif message == RESET_MESSAGE:
            size_bytes = record_log.read(4)
            size, _ = _from_bytes(int, size_bytes)
            action_bytes = size_bytes + record_log.read(cast(int, size))
            episode_seed = int.from_bytes(record_log.read(8), byteorder="little", signed=True)
            world_path = record_log.readline()
            starting_hit_points_bytes = record_log.read(4)
            starting_hit_points = _from_bytes(float, starting_hit_points_bytes)
            yield cast(
                _ResetMessage,
                (message, action_from_bytes(action_bytes), episode_seed, world_path, starting_hit_points),
            )
        elif message == LOAD_SNAPSHOT_MESSAGE:
            snapshot_path = record_log.readline()
            yield cast(_LoadSnapshotMessage, (message, snapshot_path))
        else:
            raise SwitchError(f"Invalid message type {message}")


@attr.s(auto_attribs=True, frozen=True, slots=True)
class GodotEnvActionLog(Generic[ActionType]):
    path: str
    selected_features: List[str]
    initial_episode_id: int
    messages: List[_RawMessage[ActionType]]

    @classmethod
    def parse(cls, action_log_path: str, action_type: Type[ActionType]) -> "GodotEnvActionLog[ActionType]":
        avaliable_features_query, select_features, initial_seed, *messages, close = cls.parse_message_log(
            action_log_path, action_type
        )
        assert avaliable_features_query[0] == QUERY_AVAILABLE_FEATURES_MESSAGE, (
            f"First logged message was {avaliable_features_query} but should always be a "
            f"QUERY_AVAILABLE_FEATURES_MESSAGE ({QUERY_AVAILABLE_FEATURES_MESSAGE})"
        )
        assert select_features[0] == SELECT_FEATURES_MESSAGE, (
            f"Second logged message was {select_features} but should always be a "
            f"SELECT_FEATURES_MESSAGE ({SELECT_FEATURES_MESSAGE})"
        )
        assert initial_seed[0] == SEED_MESSAGE, (
            f"Third logged message was {initial_seed} but should always be a "
            f"SELECT_FEATURES_MESSAGE ({SEED_MESSAGE})"
        )
        if close[0] != CLOSE_MESSAGE:
            logger.warning(f"{action_log_path}'s last logged message {close} is not a CLOSE_MESSAGE")
            messages.append(close)
        return cls(action_log_path, select_features[1], initial_seed[1], messages)

    @classmethod
    def parse_message_log(cls, action_log_path: str, action_type: Type[ActionType]) -> List[_RawMessage[ActionType]]:
        action_parser: Callable[[bytes], ActionType] = action_type.from_bytes
        with open(action_log_path, _BINARY_READ) as log:
            return list(_parse_raw_message_log(log, action_parser))
