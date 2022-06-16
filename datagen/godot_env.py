import math
import os
import re
import struct
import tarfile
import time
import uuid
from collections import OrderedDict
from collections import deque
from io import BufferedReader
from io import BufferedWriter
from math import prod
from pathlib import Path
from threading import Thread
from types import TracebackType
from typing import Any
from typing import Callable
from typing import Deque
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Literal
from typing import Optional
from typing import Protocol
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from typing import cast

import attr
import gym
import numpy as np
import numpy.typing as npt
import torch
from gym import spaces
from gym.spaces import Box
from loguru import logger
from numpy import typing as npt
from sentry_sdk import capture_exception

from common.errors import SwitchError
from common.utils import only
from datagen.errors import GodotError
from datagen.generate import InteractiveGodotProcess
from datagen.generate import get_first_run_action_record_path
from datagen.godot_generated_types import ACTION_MESSAGE
from datagen.godot_generated_types import CLOSE_MESSAGE
from datagen.godot_generated_types import QUERY_AVAILABLE_FEATURES_MESSAGE
from datagen.godot_generated_types import RENDER_MESSAGE
from datagen.godot_generated_types import RESET_MESSAGE
from datagen.godot_generated_types import SEED_MESSAGE
from datagen.godot_generated_types import SELECT_FEATURES_MESSAGE
from datagen.godot_generated_types import SimSpec

# Mapping of feature name to (data_type, shape).
from datagen.world_creation.constants import AvalonTask
from datagen.world_creation.constants import AvalonTaskGroup
from datagen.world_creation.world_generator import BlockingWorldGenerator
from datagen.world_creation.world_generator import GenerateWorldParams
from datagen.world_creation.world_generator import WorldGenerator

_FeatureSpecDict = OrderedDict[str, Tuple[int, Tuple[int, ...]]]

# Data dictionary of features, interpretable by a `_FeatureSpecDict`
_FeatureDataDict = Dict[str, npt.NDArray]

GODOT_ERROR_LOG_PATH = "/mnt/private/godot"


class ActionProtocol(Protocol):
    @classmethod
    def to_gym_space(cls) -> spaces.Space:
        ...

    @classmethod
    def from_input(cls: Type["ActionType"], input_vec: npt.NDArray) -> "ActionType":
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
    def from_input(cls: Type["ActionType"], input_vec: np.ndarray) -> "ActionType":
        input_vec[2:] = 0.0  # simplify action space
        return cls(*tuple(x.item() for x in input_vec))

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


class ObservationProtocol(Protocol):
    @classmethod
    def get_space_for_attribute(cls, feature: str, data_type: int, shape: Tuple[int, ...]) -> Optional[spaces.Space]:
        ...

    @classmethod
    def get_exposed_features(cls) -> Iterable[str]:
        """Features that should be exposed to the agent"""

    @classmethod
    def get_selected_features(cls) -> Iterable[str]:
        """Features to select and return from godot every step"""


ObservationType = TypeVar("ObservationType", bound=ObservationProtocol)


class AttrsObservation(ObservationProtocol):
    @classmethod
    def get_selected_features(cls) -> Tuple[str, ...]:
        return tuple(field.name for field in attr.fields(cls))


class InvalidObservationType(Exception):
    pass


ObsType = TypeVar("ObsType")


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class GoalProgressResult:
    reward: float
    is_done: bool
    log: Dict[str, Any]


class GoalEvaluator(Generic[ObsType]):
    def calculate_next_is_done_and_reward(self, observation: ObsType) -> Tuple[bool, float]:
        result = self.calculate_goal_progress(observation)
        return result.is_done, result.reward

    def calculate_goal_progress(self, observation: ObsType) -> GoalProgressResult:
        raise NotImplementedError()

    def reset(self, observation: ObsType, world_params: Optional[GenerateWorldParams] = None) -> None:
        raise NotImplementedError()


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
        check_period_seconds: float = 1.0,
        default_timeout: float = 2.0,
    ):
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

    def _kill_if_blocked(self):
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


class _GodotEnvBridge(Generic[ActionType]):
    """Low-level byte bridge to Godot leveraging unix pipes."""

    def __init__(
        self,
        action_pipe: BufferedWriter,
        action_record_pipe: BufferedWriter,
        observation_pipe: BufferedReader,
        kill_switch: _BridgeKillSwitch,
        close_timeout_seconds: float,
        screen_shape: Tuple[int, int, int],
    ):
        self._action_pipe = action_pipe
        self._action_record_pipe = action_record_pipe
        self._observation_pipe = observation_pipe
        self._kill_switch = kill_switch
        self._close_timeout_seconds = close_timeout_seconds

        self._screen_shape = screen_shape
        self._selected_features: _FeatureSpecDict = OrderedDict()

    @classmethod
    def build_by_starting_process(
        cls,
        process: InteractiveGodotProcess,
        screen_resolution: Tuple[int, int],
        build_timeout_seconds: float = 15.0,
        close_timeout_seconds: float = 15.0,
    ) -> "_GodotEnvBridge":
        """Create pipes while starting process so that the bridge doesn't block.

        Waits for godot to send a ready signal before returning
        """
        kill_switch = _BridgeKillSwitch(process, check_period_seconds=3.0, default_timeout=15.0)
        with kill_switch.watch_blocking_action(build_timeout_seconds):
            _mkpipe(process.action_pipe_path)
            _mkpipe(process.observation_pipe_path)

            process.start()

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
    def is_open(self):
        return not (self._action_pipe.closed or self._observation_pipe.closed or self._action_record_pipe.closed)

    def query_available_features(self) -> _FeatureSpecDict:
        with self._kill_switch.watch_blocking_action():
            self._send_message(QUERY_AVAILABLE_FEATURES_MESSAGE, bytes())

            feature_count = self._read_int()
            available_features: _FeatureSpecDict = OrderedDict()
            for _ in range(feature_count):
                feature_name_len = self._read_int()
                feature_name = self._observation_pipe.read(feature_name_len).decode("UTF-8")
                data_type = self._read_int()
                dim_count = self._read_int()
                dims = tuple(self._read_int() for x in range(dim_count))
                available_features[feature_name] = (data_type, dims)
            return available_features

    def select_and_cache_features(self, selected_features: _FeatureSpecDict):
        self._selected_features = selected_features
        with self._kill_switch.watch_blocking_action():
            size_doubleword = (len(selected_features)).to_bytes(4, byteorder="little", signed=False)
            feature_names_bytes = ("\n".join(selected_features.keys()) + "\n").encode("UTF-8")
            self._send_message(SELECT_FEATURES_MESSAGE, size_doubleword + feature_names_bytes)

    def seed(self, seed: int, video_id: int):
        with self._kill_switch.watch_blocking_action():
            message_bytes = seed.to_bytes(8, byteorder="little", signed=True)
            message_bytes += video_id.to_bytes(8, byteorder="little", signed=True)
            self._send_message(SEED_MESSAGE, message_bytes)

    def reset(self, null_action: ActionType, level: str, starting_hit_points: float) -> _FeatureDataDict:
        with self._kill_switch.watch_blocking_action():
            self._send_message(
                RESET_MESSAGE,
                (level + "\n").encode("UTF-8") + _to_bytes(float, starting_hit_points) + null_action.to_bytes(),
            )
            return self._read_features(self._selected_features)

    def act(self, action: ActionType) -> _FeatureDataDict:
        with self._kill_switch.watch_blocking_action():
            self._send_message(ACTION_MESSAGE, action.to_bytes())
            return self._read_features(self._selected_features)

    def render(self) -> npt.NDArray:
        with self._kill_switch.watch_blocking_action():
            self._send_message(RENDER_MESSAGE, bytes())
            return self._read_shape(self._screen_shape, prod(self._screen_shape), dtype=np.uint8)

    def close(self) -> None:
        with self._kill_switch.watch_blocking_action(self._close_timeout_seconds):
            self._send_message(CLOSE_MESSAGE, bytes())

    def after_close(self) -> None:
        self._action_pipe.close()
        self._observation_pipe.close()
        self._action_record_pipe.close()

    def _send_message(self, message_type: int, message: bytes):
        """Send a message down the godot action pipe.

        This is always a single byte message_type, followed by the message.
        """
        assert isinstance(message, bytes), f"action {cast(Any, message)} must be bytes"

        self._action_pipe_write((message_type).to_bytes(1, byteorder="little", signed=False))
        self._action_pipe_write(message)

        self._action_pipe.flush()
        self._action_record_pipe.flush()

    def _action_pipe_write(self, data: bytes):
        self._action_pipe.write(data)
        self._action_record_pipe.write(data)

    def _read_int(self) -> int:
        return cast(int, struct.unpack("i", self._observation_pipe.read(4))[0])

    def _read_shape(
        self,
        shape: Tuple[int, ...],
        size: Optional[int] = None,
        dtype: npt.DTypeLike = np.uint8,
    ) -> npt.NDArray:
        byte_buffer = self._observation_pipe.read(size if size is not None else prod(shape))
        return np.ndarray(shape=shape, dtype=dtype, buffer=byte_buffer)

    def _read_features(self, selected_features: _FeatureSpecDict) -> _FeatureDataDict:
        feature_data: _FeatureDataDict = {}
        for feature_name, (data_type, dims) in selected_features.items():
            size = prod(dims)
            if data_type != FAKE_TYPE_IMAGE:
                size = size * 4
            feature_data[feature_name] = self._read_shape(dims, size, dtype=NP_DTYPE_MAP[data_type])
        return feature_data


class GodotObservationContext(Generic[ObservationType]):
    def __init__(
        self,
        observation_type: Type[ObservationType],
        is_space_flattened: bool,
        available_features: _FeatureSpecDict,
    ):
        self.observation_type = observation_type
        self.is_space_flattened = is_space_flattened
        self.available_features = available_features
        self.selected_features = self._select_features()
        self.observation_space: Union[spaces.Dict, spaces.Space] = self._create_observation_space()
        self.flattened_observation_keys: List[str] = []
        if is_space_flattened:
            (
                self.observation_space,
                self.flattened_observation_keys,
            ) = self._flatten_observation_space(self.observation_space)

    def _select_features(self) -> _FeatureSpecDict:
        # validate our ObservationType and get selected_features
        selected_features: _FeatureSpecDict = OrderedDict()
        for field in self.observation_type.get_selected_features():
            type_and_dims = self.available_features.get(field, None)
            if type_and_dims is None:
                raise InvalidObservationType(
                    f"Could not find requested feature '{field}'. Available features are: {list(self.available_features)}"
                )
            # TODO: check that the types line up
            selected_features[field] = type_and_dims
        return selected_features

    def _create_observation_space(self) -> spaces.Dict:
        # create our observation space
        observation_space_dict = {}
        exposed_features = self.observation_type.get_exposed_features()
        for feature_name, (data_type, dims) in self.available_features.items():
            if feature_name in exposed_features:
                if feature_name not in self.selected_features:
                    raise InvalidObservationType(f"Cannot expose feature {feature_name}!")
                space = self.observation_type.get_space_for_attribute(feature_name, data_type, dims)
                if space is None:
                    space = _get_default_space(data_type, dims)
                observation_space_dict[feature_name] = space

        return spaces.Dict(observation_space_dict)

    def _flatten_observation_space(self, observation_space: spaces.Dict):
        flattened_keys = sorted(list(observation_space.spaces.keys()))
        return flatten_observation_space(observation_space, flattened_keys), flattened_keys

    def _flatten_observation(self, observation: ObservationType) -> np.ndarray:
        return flatten_observation(observation, self.flattened_observation_keys)

    def lamify(self, observation: ObservationType):
        """Convert a well-typed observation to an Env-compliant observation space dict."""
        if self.is_space_flattened:
            return self._flatten_observation(observation)

        assert isinstance(self.observation_space, spaces.Dict)
        return {x: getattr(observation, x) for x in self.observation_space.spaces.keys()}

    def make_observation(self, feature_data: Dict[str, Any]) -> ObservationType:
        return self.observation_type(**feature_data)


class GodotEnv(gym.Env, Generic[ObservationType, ActionType]):
    """An OpenAI Gym Env  that communicates with Godot over unix pipes.

    We expose the main gym.Env methods, although reset does not accept all arguments yet:
        step
        reset
        render
        close
        seed

    There are also two better-typed alternatives to step and render, act and reset_nicely,
    which accept and return the supplied ActionType and ObservationType.

    The python process must communicate in lock-step with godot, which can naturally be
    delicate and result in deadlock.
    """

    # the gym.Env#step interface returns an observation, which could
    # be something other than the screen state itself.
    #
    # To provide this affordance in the future, we could add observation shape to our SimSpec
    # and double tap the pipe on the godot side with ...observation,screen...
    # agent_observes_screen = True

    _action_pipe: BufferedWriter
    _screen_pipe: BufferedReader
    # TODO numpy array
    _current_screen: Optional[npt.NDArray]  # numpy.ndarray(x, y, 3)
    _screen_buffer_size: int
    _latest_screen: Optional[npt.NDArray]

    def __init__(
        self,
        config: SimSpec,
        observation_type: Type[ObservationType],
        action_type: Type[ActionType],
        goal_evaluator: GoalEvaluator[ObservationType],
        gpu_id: int,
        is_error_log_checked_after_each_step: bool = True,
        is_observation_space_flattened: bool = False,
        is_godot_restarted_on_error: bool = False,
    ):
        self.config = config
        self.action_type = action_type
        self.goal_evaluator = goal_evaluator

        self.action_space = self.action_type.to_gym_space()

        self.is_godot_restarted_on_error = is_godot_restarted_on_error
        self.is_reset_called_already: bool = False  # :(
        self._latest_screen: Optional[npt.NDArray] = None

        self.is_error_log_checked_after_each_step = is_error_log_checked_after_each_step

        self.episode_tracker = _EpisodeTracker(config)

        assert isinstance(self.config, SimSpec), "cannot establish godot pipe without fixed resolution"

        # TODO: this code is getting pretty duplicated and bad...
        self.world_generator = self._create_world_generator()

        self.gpu_id = gpu_id
        self.process = InteractiveGodotProcess(self.config, gpu_id=self.gpu_id)
        self._bridge: _GodotEnvBridge[ActionType] = _GodotEnvBridge.build_by_starting_process(
            self.process,
            screen_resolution=(self.config.recording_options.resolution_x, self.config.recording_options.resolution_y),
        )

        self.observation_context = GodotObservationContext(
            observation_type=observation_type,
            is_space_flattened=is_observation_space_flattened,
            available_features=self._bridge.query_available_features(),
        )
        self._bridge.select_and_cache_features(self.observation_context.selected_features)
        self.seed_nicely(self.config.random_int, -1)
        self._recent_levels: Deque[GenerateWorldParams] = deque()

    def _create_world_generator(self) -> WorldGenerator:
        return BlockingWorldGenerator(
            output_path=Path("/tmp/level_gen"), seed=2, start_difficulty=0, task_groups=(AvalonTaskGroup.ONE,)
        )

    def _restart_godot_quietly(self):
        if self.is_running:
            if self._bridge.is_open:
                self._bridge.close()
            if not self.process.is_closed:
                self.process.close(kill=True, raise_logged_errors=False)
            self._bridge.after_close()
        self.process = InteractiveGodotProcess(self.config, gpu_id=self.gpu_id)
        self._bridge = _GodotEnvBridge[ActionType].build_by_starting_process(
            self.process, (self.config.recording_options.resolution_x, self.config.recording_options.resolution_y)
        )

    def _restart_process(self, rebuild_observation_context: bool = False):
        if self.is_running:
            self.close()
        self.world_generator = self._create_world_generator()
        self.process = InteractiveGodotProcess(self.config, gpu_id=self.gpu_id)
        self._bridge = _GodotEnvBridge[ActionType].build_by_starting_process(
            self.process, (self.config.recording_options.resolution_x, self.config.recording_options.resolution_y)
        )
        if rebuild_observation_context:
            self.observation_context = GodotObservationContext(
                observation_type=self.observation_context.observation_type,
                is_space_flattened=self.observation_context.is_space_flattened,
                available_features=self._bridge.query_available_features(),
            )

    @property
    def flattened_observation_keys(self) -> List[str]:
        return self.observation_context.flattened_observation_keys

    @property
    def observation_space(self):
        return self.observation_context.observation_space

    @property
    def is_running(self):
        return self.process.is_running and self._bridge.is_open

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            seed = _randint_of_size(np.int64)
        video_id = _randint_of_size(np.int64)
        return self.seed_nicely(seed, video_id)

    def seed_nicely(self, seed: int, video_id: int):
        self._latest_seed = seed
        return self._bridge.seed(seed, video_id)

    def step(self, action: npt.NDArray):
        observation, goal_progress = self.act(self.action_type.from_input(action))
        lame_observation = self.observation_context.lamify(observation)
        return lame_observation, goal_progress.reward, goal_progress.is_done, goal_progress.log

    def act(self, action: ActionType) -> Tuple[ObservationType, GoalProgressResult]:
        """Same as `step` with observations in the ObservationType format."""
        assert isinstance(action, self.action_type), f"Must pass `{self.action_type}` objects to step"
        assert self.is_reset_called_already, "Must call reset() before calling step() or act()"
        # TODO: probably put this back, make sufficiently precise and condition on is_action_shape_checked
        # assert self.action_space.contains(attr.asdict(action)), f"Invalid action: {action}"

        feature_data = self._bridge.act(action)
        observation = self._read_observation_reply(feature_data)

        goal_progress = self.goal_evaluator.calculate_goal_progress(observation)
        self.episode_tracker.step_count_for_current_episode += 1
        return observation, goal_progress

    def _read_observation_reply(self, feature_data: _FeatureDataDict) -> ObservationType:
        self._latest_screen = feature_data["rgb"] if "rgb" in self.observation_context.selected_features else None

        if self.is_error_log_checked_after_each_step:
            self._check_for_errors_and_collect_artifacts()

        return self.observation_context.make_observation(feature_data)

    def render(self, mode: str = "rgb_array") -> npt.NDArray:
        assert mode == "rgb_array", "only rgb_array rendering is currently supported"
        if self._latest_screen is None:
            self._latest_screen = self._bridge.render()
        return self._latest_screen

    def reset(self):
        observation = self.reset_nicely()
        lame_observation = self.observation_context.lamify(observation)
        return lame_observation

    def reset_nicely(self, *, world_id: Optional[int] = None) -> ObservationType:
        self._latest_screen = None
        is_first_reset = not self.is_reset_called_already
        self.is_reset_called_already = True

        if world_id is not None:
            # reset will increment video_id on the godot side, so we decrement by one before calling seed
            self.seed_nicely(self._latest_seed, world_id - 1)

        world_params = self._get_world_params_by_id(world_id)

        # save in case we have an error
        self._recent_levels.append(world_params)
        if len(self._recent_levels) > 5:
            self._recent_levels.popleft()

        self._check_for_errors_and_collect_artifacts()

        # TODO agent passes back the world path while human player advance with world id
        world_path = f"{world_params.output}/main.tscn"
        initial_state_features = self._bridge.reset(
            self.action_type.get_null_action(), world_path, world_params.starting_hit_points
        )
        observation = self._read_observation_reply(initial_state_features)

        if not is_first_reset:
            self.episode_tracker.adjust_filename_frame_counts_and_complete_episode()

        self.goal_evaluator.reset(observation, world_params)
        return observation

    # note: this is for human playback
    def reset_nicely_with_specific_world(
        self, *, seed: int, world_id: int, world_path: str, starting_hit_points: float = 1.0
    ) -> ObservationType:
        self._latest_screen = None
        is_first_reset = not self.is_reset_called_already
        self.is_reset_called_already = True

        self.seed_nicely(seed, world_id)

        initial_state_features = self._bridge.reset(
            self.action_type.get_null_action(), world_path, starting_hit_points
        )
        observation = self._read_observation_reply(initial_state_features)

        self.process.check_for_errors()

        if not is_first_reset:
            self.episode_tracker.adjust_filename_frame_counts_and_complete_episode()

        # TODO actually get world params, maybe a `get_world_params_by_level_path`
        self.goal_evaluator.reset(observation, world_params=None)
        return observation

    def close(self):
        if self._bridge.is_open:
            self._bridge.close()
        if not self.process.is_closed:
            self.process.close(kill=False)
        self._bridge.after_close()
        self.episode_tracker.wait_for_last_episode_adjustment()

    def get_action_log(self) -> "GodotEnvActionLog[ActionType]":
        return GodotEnvActionLog.parse(self.process.action_record_path, self.action_type)

    def _check_for_errors_and_collect_artifacts(self):
        try:
            try:
                self.process.check_for_errors()
            except GodotError as ge:
                capture_exception(ge)
                os.makedirs(GODOT_ERROR_LOG_PATH, exist_ok=True)
                tar_path = f"{GODOT_ERROR_LOG_PATH}/{uuid.uuid4()}.tar"
                with tarfile.open(tar_path, "x:gz") as f:
                    f.add(self.process.action_record_path)
                    f.add(self.process.log_path)
                    f.add(self.process.config_path)
                    for level in self._recent_levels:
                        f.add(level.output)
                logger.warning(f"Godot failed! Saved recent godot levels and logs to {tar_path}")
                if self.is_godot_restarted_on_error:
                    logger.warning("Restarting godot!")
                    self._restart_godot_quietly()
                else:
                    raise
        except Exception as e:
            capture_exception(e)

    def _spawn_fresh_env(self) -> "GodotEnv[ObservationType, ActionType]":
        "Spawns a new GodotEnv with the same initial arguments as this one."
        return GodotEnv(
            self.config,
            self.observation_context.observation_type,
            self.action_type,
            self.goal_evaluator,
            self.is_error_log_checked_after_each_step,
            self.observation_context.is_space_flattened,
        )

    def get_replay(self) -> "GodotEnvReplay":
        if not self.is_reset_called_already:
            logger.warning(
                f"{self}.get_replay creates a new GodotEnv, but this env is not started. "
                f"Consider using GodotEnvReplay.from_env instead."
            )
        return GodotEnvReplay.from_env(self._spawn_fresh_env())

    def _get_world_params_by_id(self, world_id: Optional[int]) -> GenerateWorldParams:
        return only(self.world_generator.generate_batch(world_id, 1))


FAKE_TYPE_IMAGE = -1
TYPE_INT = 2
TYPE_REAL = 3
TYPE_VECTOR2 = 5
TYPE_VECTOR3 = 7

NP_DTYPE_MAP: Dict[int, Type[np.number]] = {
    TYPE_INT: np.int32,
    TYPE_REAL: np.float32,
    TYPE_VECTOR2: np.float32,
    TYPE_VECTOR3: np.float32,
    FAKE_TYPE_IMAGE: np.uint8,
}


class _EpisodeTracker:
    def __init__(self, config: SimSpec):
        self._config = config
        self._file_renaming_thread = None
        self.episode_count = 0
        self.step_count_for_current_episode = 0

    @property
    def _current_episode_folder(self):
        return os.path.join(self._config.get_dir_root(), f"{self.episode_count:06d}")

    def adjust_filename_frame_counts_and_complete_episode(self):
        if self._file_renaming_thread is not None:
            self._file_renaming_thread.join()
            self._file_renaming_thread = None

        self._file_renaming_thread = Thread(
            target=_rename_raw_files,
            args=(
                self._current_episode_folder,
                # match frame_max in a segment or end of filename,
                f"_{self._config.frame_max}(_|\\.)",
                f"_{self.step_count_for_current_episode}\\1",
            ),
        )
        self._file_renaming_thread.start()
        self.episode_count += 1
        self.step_count_for_current_episode = 0

    def wait_for_last_episode_adjustment(self):
        if self.step_count_for_current_episode > 0:
            self.adjust_filename_frame_counts_and_complete_episode()
        if self._file_renaming_thread is not None:
            self._file_renaming_thread.join()


def _mkpipe(pipe: str):
    if os.path.exists(pipe):
        os.remove(pipe)
    os.mkfifo(pipe)


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


def _randint_of_size(dtype: Type[np.integer]):
    bounds = np.iinfo(dtype)
    return np.random.randint(low=bounds.min, high=bounds.max + 1)


def _get_default_space(data_type: int, dims: Tuple[int, ...]) -> spaces.Space:
    np_dtype = NP_DTYPE_MAP[data_type]
    if data_type == FAKE_TYPE_IMAGE:
        return spaces.Box(low=0, high=255, shape=dims, dtype=np_dtype)
    return spaces.Box(low=-np.inf, high=np.inf, shape=dims, dtype=np_dtype)


def _rename_raw_files(folder: str, find_pattern: str, replace_str: str):
    for feature_file in Path(folder).glob("*.raw"):
        new_file_name = re.sub(find_pattern, replace_str, feature_file.name, 1)
        feature_file.rename(feature_file.parent / new_file_name)


def flatten_observation_space(observation_space: spaces.Dict, flattened_observation_keys: List[str]) -> spaces.Space:
    lows, highs, dtypes = [], [], set()
    for key in flattened_observation_keys:
        space = observation_space[key]
        lows.append(np.zeros(space.shape, dtype=space.dtype) + space.low)
        highs.append(np.zeros(space.shape, dtype=space.dtype) + space.high)
        dtypes.add(space.dtype)
    dtype = only(dtypes)
    flat_lows = np.concatenate(lows, axis=-1)
    flat_highs = np.concatenate(highs, axis=-1)
    shape = flat_lows.shape
    return spaces.Box(low=flat_lows, high=flat_highs, shape=shape, dtype=dtype)


def flatten_observation(observation: ObservationType, flattened_observation_keys: List[str]) -> np.ndarray:
    return np.concatenate([getattr(observation, x) for x in flattened_observation_keys], axis=-1)


_NoPayloadMessageTypes = Literal[2, 5, 6]
_NoPayloadMessage = Tuple[_NoPayloadMessageTypes]
_SeedMessage = Tuple[Literal[1], int, int]
_SelectFeaturesMessage = Tuple[Literal[4], List[str]]
_ActionMessage = Tuple[Literal[3, 0], ActionType]
_ResetMessage = Tuple[str, Literal[3, 0], ActionType]
_RawMessage = Union[
    _NoPayloadMessage,
    _SeedMessage,
    _SelectFeaturesMessage,
    _ActionMessage[ActionType],
    _ResetMessage[ActionType],
]
_no_payload_messages: Set[_NoPayloadMessageTypes] = {RENDER_MESSAGE, QUERY_AVAILABLE_FEATURES_MESSAGE, CLOSE_MESSAGE}


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
            seed = int.from_bytes(record_log.read(8), byteorder="little", signed=True)
            video_id = int.from_bytes(record_log.read(8), byteorder="little", signed=True)
            yield cast(_SeedMessage, (message, seed, video_id))

        elif message == SELECT_FEATURES_MESSAGE:
            count = int.from_bytes(record_log.read(4), byteorder="little", signed=False)
            feature_names = list(record_log.readline().decode("UTF-8")[:-1] for _ in range(count))
            yield cast(_SelectFeaturesMessage, (message, feature_names))

        elif message == ACTION_MESSAGE:
            size_bytes = record_log.read(4)
            size, _ = _from_bytes(int, size_bytes)
            action_bytes = size_bytes + record_log.read(cast(int, size))
            yield cast(_ActionMessage, (message, action_from_bytes(action_bytes)))

        elif message == RESET_MESSAGE:
            level_name = record_log.readline()
            size_bytes = record_log.read(4)
            size, _ = _from_bytes(int, size_bytes)
            action_bytes = size_bytes + record_log.read(cast(int, size))
            yield cast(_ResetMessage, (message, level_name, action_from_bytes(action_bytes)))
        else:
            raise SwitchError(f"Invalid message type {message}")


@attr.s(auto_attribs=True, frozen=True, slots=True)
class GodotEnvActionLog(Generic[ActionType]):
    path: str
    selected_features: List[str]
    initial_seed: int
    initial_video_id: int
    messages: List[_RawMessage[ActionType]]

    @classmethod
    def parse(cls, action_log_path: str, action_type: Type[ActionType]) -> "GodotEnvActionLog[ActionType]":
        action_parser: Callable[[bytes], ActionType] = action_type.from_bytes
        with open(action_log_path, _BINARY_READ) as log:
            avaliable_features_query, select_features, initial_seed, *messages, close = _parse_raw_message_log(
                log, action_parser
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
            # TODO this is not always true anymore with human playback
            # assert initial_seed[2] == 0, f"Initial seed video_id {initial_seed[2]} != 0"
            if close[0] != CLOSE_MESSAGE:
                logger.warning(f"{action_log_path}'s last logged message {close} is not a CLOSE_MESSAGE")
                messages.append(close)
            return cls(action_log_path, select_features[1], initial_seed[1], initial_seed[2], messages)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class GodotEnvReplay(Generic[ObservationType, ActionType]):
    env: GodotEnv[ObservationType, ActionType]
    action_log: GodotEnvActionLog[ActionType]

    @classmethod
    def from_env(cls, env: GodotEnv[ObservationType, ActionType]) -> "GodotEnvReplay":
        action_record_path = str(get_first_run_action_record_path(env.process.config_path))
        log = GodotEnvActionLog.parse(action_record_path, env.action_type)
        return cls(env, log)

    def __attrs_post_init__(self):
        env = self.env
        action_log = self.action_log
        assert not env.is_reset_called_already, f"{env} looks like it has already been run."
        assert env.config.random_int == action_log.initial_seed, (
            f"{env}.config.random_int {env.config.random_int} does not match "
            f"logged seed {action_log.initial_seed} from {action_log.path}"
        )
        selected_feature_names = list(env.observation_context.selected_features.keys())
        assert selected_feature_names == action_log.selected_features, (
            f"{env}'s selected_features names {selected_feature_names} does not match "
            f"{action_log.selected_features} logged in {action_log.path}"
        )

    def __call__(self) -> Iterator[Union[Tuple[ObservationType, Optional[GoalProgressResult]], npt.NDArray]]:
        env = self.env
        action_log = self.action_log
        for message in action_log.messages:
            if message[0] == ACTION_MESSAGE:
                action = message[1]
                yield env.act(action)
            elif message[0] == RESET_MESSAGE:
                yield (env.reset_nicely(), message[1], None)
            elif message[0] == RENDER_MESSAGE:
                yield env.render()
            elif message[0] == SEED_MESSAGE:
                seed, video_id = message[1:]
                yield env.seed_nicely(seed, video_id)
            else:
                raise SwitchError(f"Invalid replay message {message}")


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class VRActionType(AttrsAction):
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
    def from_input(cls, input_dict: Dict[str, np.ndarray]) -> "VRActionType":
        # clipping each triplet to sphere
        input_real = input_dict["real"]
        if isinstance(input_real, torch.Tensor):
            input_real = input_real.numpy()
        triplet_norm = np.linalg.norm(np.reshape(input_real, (6, 3)), axis=-1)
        scale = 1 / np.clip(triplet_norm, a_min=1, a_max=float("inf"))
        clipped_real = np.repeat(scale, 3) * input_real
        input_vec = np.concatenate([clipped_real, input_dict["discrete"]], axis=-1)
        action = cls(*tuple(x.item() for x in input_vec))
        return action


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class MouseKeyboardActionType(AttrsAction):
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
    def from_input(cls, input_dict: Dict[str, np.ndarray]) -> "MouseKeyboardActionType":
        clipped_real = np.clip(input_dict["real"], a_min=-1, a_max=1)
        input_vec = np.concatenate([clipped_real, input_dict["discrete"]], axis=-1)

        return cls(*tuple(x.item() for x in input_vec))


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class AvalonObservationType(AttrsObservation):
    rgbd: npt.NDArray[np.uint8]
    video_id: npt.NDArray[np.int32]
    frame_id: npt.NDArray[np.int32]
    reward: npt.NDArray[np.float32]
    is_done: npt.NDArray[np.float32]
    is_dead: npt.NDArray[np.float32]

    physical_body_position: npt.NDArray[np.float32]
    physical_head_position: npt.NDArray[np.float32]
    physical_left_hand_position: npt.NDArray[np.float32]
    physical_right_hand_position: npt.NDArray[np.float32]
    physical_body_rotation: npt.NDArray[np.float32]
    physical_head_rotation: npt.NDArray[np.float32]
    physical_left_hand_rotation: npt.NDArray[np.float32]
    physical_right_hand_rotation: npt.NDArray[np.float32]
    physical_body_delta_position: npt.NDArray[np.float32]
    physical_head_delta_position: npt.NDArray[np.float32]
    physical_left_hand_delta_position: npt.NDArray[np.float32]
    physical_right_hand_delta_position: npt.NDArray[np.float32]
    physical_body_delta_rotation: npt.NDArray[np.float32]
    physical_head_delta_rotation: npt.NDArray[np.float32]
    physical_left_hand_delta_rotation: npt.NDArray[np.float32]
    physical_right_hand_delta_rotation: npt.NDArray[np.float32]
    physical_head_relative_position: npt.NDArray[np.float32]
    physical_left_hand_relative_position: npt.NDArray[np.float32]
    physical_right_hand_relative_position: npt.NDArray[np.float32]
    physical_head_relative_rotation: npt.NDArray[np.float32]
    physical_left_hand_relative_rotation: npt.NDArray[np.float32]
    physical_right_hand_relative_rotation: npt.NDArray[np.float32]
    left_hand_thing_colliding_with_hand: npt.NDArray[np.float32]
    left_hand_held_thing: npt.NDArray[np.float32]
    right_hand_thing_colliding_with_hand: npt.NDArray[np.float32]
    right_hand_held_thing: npt.NDArray[np.float32]

    nearest_food_position: npt.NDArray[np.float32]
    nearest_food_id: npt.NDArray[np.float32]
    is_food_present_in_world: npt.NDArray[np.float32]

    physical_body_kinetic_energy_expenditure: npt.NDArray[np.float32]
    physical_body_potential_energy_expenditure: npt.NDArray[np.float32]
    physical_head_potential_energy_expenditure: npt.NDArray[np.float32]
    physical_left_hand_kinetic_energy_expenditure: npt.NDArray[np.float32]
    physical_left_hand_potential_energy_expenditure: npt.NDArray[np.float32]
    physical_right_hand_kinetic_energy_expenditure: npt.NDArray[np.float32]
    physical_right_hand_potential_energy_expenditure: npt.NDArray[np.float32]
    fall_damage: npt.NDArray[np.float32]
    total_energy_expenditure: npt.NDArray[np.float32]
    hit_points_lost_from_enemies: npt.NDArray[np.float32]
    hit_points_gained_from_eating: npt.NDArray[np.float32]
    hit_points: npt.NDArray[np.float32]

    @classmethod
    def get_space_for_attribute(cls, feature: str, data_type: int, shape: Tuple[int, ...]) -> Optional[spaces.Space]:
        return None

    @classmethod
    def get_exposed_features(cls) -> Tuple[str, ...]:
        return (
            "rgbd",
            "physical_body_delta_position",
            "physical_body_delta_rotation",
            "physical_head_delta_position",
            "physical_left_hand_delta_position",
            "physical_right_hand_delta_position",
            "physical_head_delta_rotation",
            "physical_left_hand_delta_rotation",
            "physical_right_hand_delta_rotation",
            "physical_head_relative_position",
            "physical_left_hand_relative_position",
            "physical_right_hand_relative_position",
            "physical_head_relative_rotation",
            "physical_left_hand_relative_rotation",
            "physical_right_hand_relative_rotation",
            "left_hand_thing_colliding_with_hand",
            "left_hand_held_thing",
            "right_hand_thing_colliding_with_hand",
            "right_hand_held_thing",
            "fall_damage",
            "total_energy_expenditure",
            "hit_points_lost_from_enemies",
            "hit_points_gained_from_eating",
            "hit_points",
        )


FRAMES_PER_MINUTE = 600


@attr.s(auto_attribs=True, collect_by_mro=True)
class GodotGoalEvaluator(GoalEvaluator[AvalonObservationType]):
    def calculate_goal_progress(self, observation: AvalonObservationType) -> GoalProgressResult:
        self.update_score(observation)

        is_done = observation.is_done.item() > 0

        truncated = False
        if not is_done and observation.frame_id.item() + 1 >= self.current_level_frame_limit():
            is_done = True
            truncated = True

        return GoalProgressResult(
            is_done=is_done,
            reward=observation.reward.item(),
            log={
                "total_energy_expenditure": observation.total_energy_expenditure.item(),
                "success": 1 if self.is_all_food_eaten else 0,
                "score": self.score,
                "difficulty": self.world_params.difficulty,
                "task": self.world_params.task.value,
                "video_id": observation.video_id.item(),
                "world_index": self.world_params.index,
                "TimeLimit.truncated": truncated,
            },
        )

    def update_score(self, observation: AvalonObservationType):
        if not self.is_all_food_eaten:
            self.score = observation.hit_points.item()
        self.is_all_food_eaten = observation.is_food_present_in_world.item() < 0.1
        if observation.is_dead.item():
            self.score = observation.hit_points.item()

    def current_level_frame_limit(self) -> int:
        if self.world_params.task in (AvalonTask.SURVIVE, AvalonTask.FIND, AvalonTask.GATHER, AvalonTask.NAVIGATE):
            return 15 * FRAMES_PER_MINUTE
        elif self.world_params.task in (AvalonTask.STACK, AvalonTask.CARRY, AvalonTask.EXPLORE):
            return 10 * FRAMES_PER_MINUTE
        else:
            return 5 * FRAMES_PER_MINUTE

    def reset(self, observation: AvalonObservationType, world_params: Optional[GenerateWorldParams] = None) -> None:
        self.world_params = world_params
        self.score = observation.hit_points.item()
        self.is_all_food_eaten = observation.is_food_present_in_world.item() < 0.1


class AvalonScoreEvaluator(GodotGoalEvaluator):
    def update_score(self, observation: AvalonObservationType):
        if not self.is_all_food_eaten:
            self.score += observation.reward.item()
            # we replace energy cost with a 1/3000 per frame cost
            # self.score += observation.total_energy_expenditure.item()
            self.score -= 1 / 3000.0
        self.is_all_food_eaten = observation.is_food_present_in_world.item() < 0.1
        if observation.is_dead.item() or self.score < 0:
            self.score = 0


@attr.s(auto_attribs=True, collect_by_mro=True)
class DynamicFrameLimitGodotGoalEvaluator(GodotGoalEvaluator):
    max_frames: int = 120
    difficulty_time_bonus_factor: float = 1

    def current_level_frame_limit(self) -> int:
        return int(self.max_frames * (10 ** (self.difficulty_time_bonus_factor * self.world_params.difficulty)))
