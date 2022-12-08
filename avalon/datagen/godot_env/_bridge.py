import math
import os
import struct
import time
from collections import OrderedDict
from io import BufferedReader
from io import BufferedWriter
from math import prod
from threading import Thread
from types import TracebackType
from typing import Any
from typing import Dict
from typing import Generic
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Type
from typing import cast

import numpy as np
from loguru import logger
from numpy import typing as npt

from avalon.datagen.errors import GodotError
from avalon.datagen.godot_env.actions import ActionType
from avalon.datagen.godot_env.actions import DebugCameraAction
from avalon.datagen.godot_env.actions import _to_bytes
from avalon.datagen.godot_env.interactive_godot_process import InteractiveGodotProcess
from avalon.datagen.godot_env.observations import FAKE_TYPE_IMAGE
from avalon.datagen.godot_env.observations import NP_DTYPE_MAP
from avalon.datagen.godot_env.observations import FeatureSpecDict
from avalon.datagen.godot_generated_types import ACTION_MESSAGE
from avalon.datagen.godot_generated_types import BRIDGE_LOG_SIGNAL
from avalon.datagen.godot_generated_types import CLOSE_MESSAGE
from avalon.datagen.godot_generated_types import DEBUG_CAMERA_ACTION_MESSAGE
from avalon.datagen.godot_generated_types import LOAD_SNAPSHOT_MESSAGE
from avalon.datagen.godot_generated_types import QUERY_AVAILABLE_FEATURES_MESSAGE
from avalon.datagen.godot_generated_types import RENDER_MESSAGE
from avalon.datagen.godot_generated_types import RESET_MESSAGE
from avalon.datagen.godot_generated_types import SAVE_SNAPSHOT_MESSAGE
from avalon.datagen.godot_generated_types import SEED_MESSAGE
from avalon.datagen.godot_generated_types import SELECT_FEATURES_MESSAGE

# Data dictionary of features, interpretable by a `FeatureSpecDict`
FeatureDataDict = Dict[str, npt.NDArray]

_BINARY_READ: Literal["br"] = "br"
_BINARY_WRITE: Literal["bw"] = "bw"


class _BridgeKillSwitch:
    """Kills the GodotProcess if a blocking action fails to complete.

    Starts a background process that will kill the godot_process if an any
    logic wrapped in a `with watch_blocking_action()` fails to complete within
    the timeout.
    """

    def __init__(
        self,
        godot_process: InteractiveGodotProcess,
        check_period_seconds: float = 5.0,
        default_timeout: float = 30.0,
    ) -> None:
        self._kill_time: float = math.inf
        self._godot_process = godot_process
        self._default_timeout = default_timeout
        self._kill_error: Optional[GodotError] = None

        self._check_period_seconds = check_period_seconds
        self._thread = Thread(
            target=self._kill_if_blocked,
            daemon=True,
        )
        self._thread.start()

    def watch_blocking_action(self, timeout_seconds: Optional[float] = None):
        self._kill_time = time.time() + (timeout_seconds or self._default_timeout)
        return self

    def __enter__(self) -> "_BridgeKillSwitch":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self._kill_time = math.inf
        if self._kill_error is None:
            return
        if exc_val is not None:
            exc_val.__cause__ = self._kill_error
            return
        raise self._kill_error

    def raise_any_logged_errors(self) -> None:
        self._godot_process.raise_any_logged_godot_errors()

    def _kill_if_blocked(self) -> None:
        while True:
            time.sleep(self._check_period_seconds)
            if self._godot_process.is_finished:
                return
            if self._godot_process.is_running and time.time() > self._kill_time:
                logger.error(f"_BridgeKillSwitch killing godot: blocking action took too long")
                try:
                    self._godot_process.check_for_errors()
                    self._kill_error = GodotError(f"_BridgeKillSwitch killing process")
                    self._godot_process.close(kill=True)
                except GodotError as e:
                    self._kill_error = GodotError(f"_BridgeKillSwitch killed process: {e.args[0]}")
                return


class GodotEnvBridge(Generic[ActionType]):
    """Low-level byte bridge to Godot leveraging unix pipes."""

    def __init__(
        self,
        action_pipe: BufferedWriter,
        action_record_pipe: BufferedWriter,
        observation_pipe: BufferedReader,
        kill_switch: _BridgeKillSwitch,
        close_timeout_seconds: float,
        screen_shape: Tuple[int, int, int],
    ) -> None:
        self._action_pipe = action_pipe
        self._action_record_pipe = action_record_pipe
        self._observation_pipe = observation_pipe
        self._kill_switch = kill_switch
        self._close_timeout_seconds = close_timeout_seconds

        self._screen_shape = screen_shape
        self._selected_features: FeatureSpecDict = OrderedDict()

    @classmethod
    def _kill_on_parsing_errors_to_avoid_zombies(cls, process: InteractiveGodotProcess):
        try:
            process.wait_for_log_signal(BRIDGE_LOG_SIGNAL)
        except (GodotError, TimeoutError):
            process.close(kill=True)
            raise

    @classmethod
    def build_by_starting_process(
        cls,
        process: InteractiveGodotProcess,
        screen_resolution: Tuple[int, int],
        build_timeout_seconds: float = 30.0,
        close_timeout_seconds: float = 30.0,
    ) -> "GodotEnvBridge":
        """Create pipes while starting process so that the bridge doesn't block.

        Waits for godot to send a ready signal before returning
        """
        kill_switch = _BridgeKillSwitch(process)
        with kill_switch.watch_blocking_action(build_timeout_seconds):
            _mkpipe(process.action_pipe_path)
            _mkpipe(process.observation_pipe_path)

            process.start()

            cls._kill_on_parsing_errors_to_avoid_zombies(process)

            bridge = cls(
                action_pipe=open(process.action_pipe_path, _BINARY_WRITE),
                action_record_pipe=open(process.action_record_path, _BINARY_WRITE),
                observation_pipe=open(process.observation_pipe_path, _BINARY_READ),
                kill_switch=kill_switch,
                close_timeout_seconds=close_timeout_seconds,
                screen_shape=(*screen_resolution, 3),
            )

            process.wait_for_ready_signal()

        return bridge

    @property
    def is_open(self) -> bool:
        return not (self._action_pipe.closed or self._observation_pipe.closed or self._action_record_pipe.closed)

    def query_available_features(self) -> FeatureSpecDict:
        with self._kill_switch.watch_blocking_action():
            self._send_message(QUERY_AVAILABLE_FEATURES_MESSAGE, bytes())

            feature_count = self._read_int()
            available_features: FeatureSpecDict = OrderedDict()
            for _ in range(feature_count):
                feature_name_len = self._read_int()
                feature_name = self._observation_pipe.read(feature_name_len).decode("UTF-8")
                data_type = self._read_int()
                dim_count = self._read_int()
                dims = tuple(self._read_int() for x in range(dim_count))
                available_features[feature_name] = (data_type, dims)
            return available_features

    def select_and_cache_features(self, selected_features: FeatureSpecDict) -> None:
        self._selected_features = selected_features
        with self._kill_switch.watch_blocking_action():
            size_doubleword = (len(selected_features)).to_bytes(4, byteorder="little", signed=False)
            feature_names_bytes = ("\n".join(selected_features.keys()) + "\n").encode("UTF-8")
            self._send_message(SELECT_FEATURES_MESSAGE, size_doubleword + feature_names_bytes)

    def seed(self, episode_seed: int) -> None:
        with self._kill_switch.watch_blocking_action():
            message_bytes = episode_seed.to_bytes(8, byteorder="little", signed=True)
            self._send_message(SEED_MESSAGE, message_bytes)

    def reset(
        self, null_action: ActionType, episode_seed: int, world_path: str, starting_hit_points: float
    ) -> FeatureDataDict:
        with self._kill_switch.watch_blocking_action():
            self._send_message(
                RESET_MESSAGE,
                null_action.to_bytes()
                + episode_seed.to_bytes(8, byteorder="little", signed=True)
                + (world_path + "\n").encode("UTF-8")
                + _to_bytes(float, starting_hit_points),
            )
            return self._read_features(self._selected_features)

    def act(self, action: ActionType) -> FeatureDataDict:
        with self._kill_switch.watch_blocking_action():
            self._send_message(ACTION_MESSAGE, action.to_bytes())
            return self._read_features(self._selected_features)

    def debug_act(self, action: DebugCameraAction) -> FeatureDataDict:
        with self._kill_switch.watch_blocking_action():
            self._send_message(DEBUG_CAMERA_ACTION_MESSAGE, action.to_bytes())
            return self._read_features(self._selected_features)

    def render(self) -> npt.NDArray:
        with self._kill_switch.watch_blocking_action():
            self._send_message(RENDER_MESSAGE, bytes())
            return self._read_shape(self._screen_shape, prod(self._screen_shape), dtype=np.uint8)

    def save_snapshot(self) -> str:
        with self._kill_switch.watch_blocking_action():
            self._send_message(SAVE_SNAPSHOT_MESSAGE, bytes())
            snapshot_path = self._observation_pipe.readline().decode("UTF-8")[:-1]
            return snapshot_path

    def load_snapshot(self, snapshot_path: str) -> FeatureDataDict:
        with self._kill_switch.watch_blocking_action():
            self._send_message(
                LOAD_SNAPSHOT_MESSAGE,
                (snapshot_path + "\n").encode("UTF-8"),
            )
            return self._read_features(self._selected_features)

    def close(self) -> None:
        with self._kill_switch.watch_blocking_action(self._close_timeout_seconds):
            self._send_message(CLOSE_MESSAGE, bytes())

    def after_close(self) -> None:
        try:
            self._action_pipe.close()
        except BrokenPipeError:
            pass
        try:
            self._observation_pipe.close()
        except BrokenPipeError:
            pass
        try:
            self._action_record_pipe.close()
        except BrokenPipeError:
            pass

    def _send_message(self, message_type: int, message: bytes) -> None:
        """Send a message down the godot action pipe.

        This is always a single byte message_type, followed by the message.
        """
        assert isinstance(message, bytes), f"action {cast(Any, message)} must be bytes"
        self._action_pipe_write((message_type).to_bytes(1, byteorder="little", signed=False))
        self._action_pipe_write(message)

        self._action_pipe.flush()
        self._action_record_pipe.flush()

    def _action_pipe_write(self, data: bytes) -> None:
        self._action_pipe.write(data)
        self._action_record_pipe.write(data)

    def _read_int(self) -> int:
        return cast(int, struct.unpack("i", self._observation_pipe.read(4))[0])

    def _read_shape(
        self,
        shape: Tuple[int, ...],
        size: Optional[int] = None,
        dtype: npt.DTypeLike = np.uint8,
        feature_name: Optional[str] = None,
    ) -> npt.NDArray:
        read_size = size if size is not None else prod(shape)
        byte_buffer = self._observation_pipe.read(read_size)
        try:
            return np.ndarray(shape=shape, dtype=dtype, buffer=byte_buffer)
        except TypeError as te:
            self._kill_switch.raise_any_logged_errors()
            raise GodotError(
                f"Invalid number of bytes for feature {feature_name}: "
                f"got {len(byte_buffer)}, expected {read_size} of {dtype}"
            ) from te

    def _read_features(self, selected_features: FeatureSpecDict) -> FeatureDataDict:
        feature_data: FeatureDataDict = {}
        for feature_name, (data_type, dims) in selected_features.items():
            size = prod(dims)
            if data_type != FAKE_TYPE_IMAGE:
                size = size * 4
            feature_data[feature_name] = self._read_shape(
                dims, size, dtype=NP_DTYPE_MAP[data_type], feature_name=feature_name
            )
        return feature_data


def _mkpipe(pipe: str) -> None:
    if os.path.exists(pipe):
        os.remove(pipe)
    os.mkfifo(pipe)
