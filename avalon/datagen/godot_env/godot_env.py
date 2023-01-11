import shutil
from collections import deque
from io import BufferedReader
from io import BufferedWriter
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Deque
from typing import Dict
from typing import Final
from typing import Generic
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
from typing import cast

import gym
import numpy as np
from loguru import logger
from numpy import typing as npt

from avalon.common.error_utils import capture_exception
from avalon.common.utils import only
from avalon.contrib.s3_utils import SimpleS3Client
from avalon.datagen.errors import GodotError
from avalon.datagen.godot_env._bridge import GodotEnvBridge
from avalon.datagen.godot_env.action_log import GodotEnvActionLog
from avalon.datagen.godot_env.actions import ActionType
from avalon.datagen.godot_env.actions import DebugCameraAction
from avalon.datagen.godot_env.goals import GoalEvaluator
from avalon.datagen.godot_env.goals import GoalProgressResult
from avalon.datagen.godot_env.interactive_godot_process import InteractiveGodotProcess
from avalon.datagen.godot_env.observations import FeatureDataDict
from avalon.datagen.godot_env.observations import GodotObservationContext
from avalon.datagen.godot_env.observations import ObservationType
from avalon.datagen.godot_generated_types import SimSpec

# Mapping of feature name to (data_type, shape).
from avalon.datagen.world_creation.constants import STARTING_HIT_POINTS
from avalon.datagen.world_creation.world_generator import EmptyLevelGenerator
from avalon.datagen.world_creation.world_generator import GeneratedWorldParamsType
from avalon.datagen.world_creation.world_generator import WorldGenerator

_DEFAULT_EPISODE_SEED: Final = 0

if TYPE_CHECKING:
    from avalon.datagen.godot_env.replay import GodotEnvReplay


class GodotEnv(gym.Env, Generic[ObservationType, ActionType, GeneratedWorldParamsType]):
    """An OpenAI Gym Env that communicates with Godot over unix pipes.

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
        goal_evaluator: GoalEvaluator[ObservationType, GeneratedWorldParamsType],
        gpu_id: int = 0,
        is_logging_artifacts_on_error_to_s3: bool = False,
        s3_bucket_name: Optional[str] = None,
        is_error_log_checked_after_each_step: bool = True,
        is_observation_space_flattened: bool = False,
        is_godot_restarted_on_error: bool = False,
        is_dev_flag_added: bool = False,
        run_uuid: Optional[str] = None,
    ) -> None:
        self.config = config
        self.action_type = action_type
        self.goal_evaluator = goal_evaluator

        self.action_space = self.action_type.to_gym_space()

        self.is_dev_flag_added = is_dev_flag_added
        self.is_logging_artifacts_on_error_to_s3 = is_logging_artifacts_on_error_to_s3
        if self.is_logging_artifacts_on_error_to_s3:
            assert s3_bucket_name is not None, "Must provide S3 bucket name"
        self.s3_bucket_name = s3_bucket_name
        self.is_godot_restarted_on_error = is_godot_restarted_on_error
        self.is_reset_called_already: bool = False  # :(
        self._latest_screen: Optional[npt.NDArray] = None
        self._latest_episode_seed: int = 0

        self.is_error_log_checked_after_each_step = is_error_log_checked_after_each_step

        assert isinstance(self.config, SimSpec), "cannot establish godot pipe without fixed resolution"

        # TODO: this code is getting pretty duplicated and bad...
        self.world_generator = self._create_world_generator()

        self.gpu_id = gpu_id
        self.process = InteractiveGodotProcess(
            self.config, is_dev_flag_added=is_dev_flag_added, gpu_id=self.gpu_id, run_uuid=run_uuid
        )
        self._bridge: GodotEnvBridge[ActionType] = GodotEnvBridge.build_by_starting_process(
            self.process,
            screen_resolution=(self.config.recording_options.resolution_x, self.config.recording_options.resolution_y),
        )

        self.observation_context = GodotObservationContext(
            observation_type=observation_type,
            is_space_flattened=is_observation_space_flattened,
            available_features=self._bridge.query_available_features(),
        )
        self._bridge.select_and_cache_features(self.observation_context.selected_features)
        self.seed_nicely(0)
        self.recent_worlds: Deque[GeneratedWorldParamsType] = deque()

    def _create_world_generator(self) -> WorldGenerator[GeneratedWorldParamsType]:
        return cast(
            WorldGenerator[GeneratedWorldParamsType], EmptyLevelGenerator(base_path=Path("/tmp/level_gen"), seed=0)
        )

    def _restart_godot_quietly(self) -> None:
        if self.is_running:
            if self._bridge.is_open:
                try:
                    self._bridge.close()
                except BrokenPipeError:
                    # Handle this error because there's a good chance that godot has crashed.
                    logger.info("caught broken pipe when restarting godot")
            if not self.process.is_closed:
                self.process.close(kill=True, raise_logged_errors=False)
            self._bridge.after_close()
        self.process = InteractiveGodotProcess(self.config, gpu_id=self.gpu_id)
        self._bridge = GodotEnvBridge[ActionType].build_by_starting_process(
            self.process, (self.config.recording_options.resolution_x, self.config.recording_options.resolution_y)
        )

    def _restart_process(self, rebuild_observation_context: bool = False) -> None:
        if self.is_running:
            self.close()
        self.world_generator = self._create_world_generator()
        self.process = InteractiveGodotProcess(self.config, gpu_id=self.gpu_id)
        self._bridge = GodotEnvBridge[ActionType].build_by_starting_process(
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
        return self.seed_nicely(seed)

    def seed_nicely(self, episode_seed: int):
        self._latest_episode_seed = episode_seed
        return self._bridge.seed(episode_seed)

    def step(self, action: Dict[str, np.ndarray]):
        observation, goal_progress = self.act(self.action_type.from_input(action))
        lame_observation = self.observation_context.lamify(observation)
        return lame_observation, goal_progress.reward, goal_progress.is_done, goal_progress.log

    def act(self, action: ActionType) -> Tuple[ObservationType, GoalProgressResult]:
        """Same as `step` with observations in the ObservationType format."""
        assert isinstance(action, self.action_type), f"Must pass `{self.action_type}` objects to step"
        assert self.is_reset_called_already, "Must call reset() before calling step() or act()"
        # TODO: probably put this back, make sufficiently precise and condition on is_action_shape_checked
        # assert self.action_space.contains(attr.asdict(action)), f"Invalid action: {action}"

        try:
            feature_data = self._bridge.act(action)
        except GodotError as e:
            # Godot error prevented reading feature data...
            self._check_for_errors_and_collect_artifacts()
            raise
        observation = self._read_observation_reply(feature_data)

        goal_progress = self.goal_evaluator.calculate_goal_progress(observation)
        return observation, goal_progress

    def debug_act(self, action: DebugCameraAction) -> ObservationType:
        assert isinstance(action, DebugCameraAction), f"Must pass `DebugCameraAction` objects to debug_act"
        feature_data = self._bridge.debug_act(action)
        return self._read_observation_reply(feature_data)

    def save_snapshot(self) -> Path:
        assert self.is_reset_called_already, "Cannot take a snapshot without starting a world (via reset)"
        return Path(self._bridge.save_snapshot())

    def load_snapshot(self, snapshot_path: Path) -> Tuple[ObservationType, GoalProgressResult]:
        assert (
            not snapshot_path.is_file()
        ), f"snapshot_path {snapshot_path} should be the path to the directory returned from save_snapshot."
        assert snapshot_path.exists(), f"cannot load snapshot_path {snapshot_path} because it doesn't exist."
        self.is_reset_called_already = True
        feature_data = self._bridge.load_snapshot(snapshot_path.as_posix())

        observation = self._read_observation_reply(feature_data)

        goal_progress = self.goal_evaluator.calculate_goal_progress(observation)
        return observation, goal_progress

    def _read_observation_reply(self, feature_data: FeatureDataDict) -> ObservationType:
        self._latest_screen = feature_data["rgb"] if "rgb" in self.observation_context.selected_features else None

        if self.is_error_log_checked_after_each_step:
            self._check_for_errors_and_collect_artifacts()

        return self.observation_context.make_observation(feature_data)

    def render(self, mode: str = "rgb_array") -> npt.NDArray:  # type: ignore[override]
        assert mode == "rgb_array", "only rgb_array rendering is currently supported"
        if self._latest_screen is None:
            self._latest_screen = self._bridge.render()
            assert type(self._latest_screen) == npt.NDArray
        return self._latest_screen

    def reset(self):
        observation = self.reset_nicely()
        lame_observation = self.observation_context.lamify(observation)
        return lame_observation

    def reset_nicely(self, *, world_id: Optional[int] = None, episode_seed: Optional[int] = None) -> ObservationType:
        self._latest_screen = None
        self.is_reset_called_already = True

        if episode_seed is None:
            episode_seed = _DEFAULT_EPISODE_SEED
        self._latest_episode_seed = episode_seed

        # calls generate batch and creates a level (from somewhere?)
        world_params = self._get_world_params_by_id(world_id)

        # save in case we have an error
        self.recent_worlds.append(world_params)
        if len(self.recent_worlds) > 5:
            self.recent_worlds.popleft()

        self._check_for_errors_and_collect_artifacts()

        initial_state_features = self._bridge.reset(
            self.action_type.get_null_action(), episode_seed, world_params.main_scene_path, STARTING_HIT_POINTS
        )
        observation = self._read_observation_reply(initial_state_features)

        self.goal_evaluator.reset(observation, world_params)
        return observation

    # note: this is for human playback
    def reset_nicely_with_specific_world(
        self,
        *,
        episode_seed: int,
        world_path: Optional[Union[str, Path]] = None,
        world_params: Optional[GeneratedWorldParamsType] = None,
        starting_hit_points: float = 1.0,
    ) -> ObservationType:
        """Specify the world with `world_params` if available (necessary for the default goal evaluator to work),
        and `world_path` if you're loading a custom non-autogenerated level.
        """
        if not world_path:
            assert world_params is not None
            world_path = world_params.output + "/main.tscn"
        world_path = Path(world_path)
        assert world_path.name.endswith(".tscn"), f"{world_path} must be a valid godot scene generated by avalon."
        assert world_path.exists(), f"No such world {world_path}"

        self._latest_screen = None
        self.is_reset_called_already = True
        self._latest_episode_seed = episode_seed

        initial_state_features = self._bridge.reset(
            self.action_type.get_null_action(), episode_seed, str(world_path), starting_hit_points
        )
        observation = self._read_observation_reply(initial_state_features)

        self.process.check_for_errors()

        if world_params:
            self.goal_evaluator.reset(observation, world_params=world_params)
        else:
            self.goal_evaluator.reset(observation, world_params=None)
        return observation

    def close(self) -> None:
        if self._bridge.is_open:
            self._bridge.close()
        if not self.process.is_closed:
            self.process.close(kill=False)
        self._bridge.after_close()
        self.world_generator.close()

    def __del__(self) -> None:
        try:
            if self.is_running:
                self.close()
        except AttributeError:
            # Ignore errors from incomplete init
            pass
        except BrokenPipeError:
            # ignore errors from race conditions if the pipe was broken by a different close
            if self.is_running:
                raise

    def cleanup(self) -> None:
        shutil.rmtree(self.config.dir_root, ignore_errors=True)

    def get_action_log(self) -> "GodotEnvActionLog[ActionType]":
        return GodotEnvActionLog.parse(self.process.action_record_path, self.action_type)

    def _check_for_errors_and_collect_artifacts(self):
        try:
            self.process.check_for_errors()
        except GodotError as ge:
            capture_exception(ge, is_thrown_without_sentry=False)

            tar_path = self.process.save_artifacts(self.recent_worlds)
            logger.warning(f"Godot failed! Saved recent godot levels and logs to {tar_path}")

            if self.is_logging_artifacts_on_error_to_s3:
                logger.warning(f"Uploading artifacts to S3 bucket {self.s3_bucket_name} with key {tar_path}.")
                s3_client = SimpleS3Client(bucket_name=self.s3_bucket_name)
                s3_client.upload_from_file(tar_path, key=Path(tar_path).name)

            if self.is_godot_restarted_on_error:
                logger.warning("Restarting godot!")
                self._restart_godot_quietly()
            else:
                raise
        except ValueError as e:
            if e.args != ("read of closed file",):
                raise e

    def _spawn_fresh_copy_of_env(
        self, run_uuid: str
    ) -> "GodotEnv[ObservationType, ActionType, GeneratedWorldParamsType]":
        "Spawns a new GodotEnv with the same initial arguments as this one."
        return GodotEnv(
            config=self.config,
            observation_type=self.observation_context.observation_type,
            action_type=self.action_type,
            goal_evaluator=self.goal_evaluator,
            gpu_id=self.gpu_id,
            is_logging_artifacts_on_error_to_s3=self.is_logging_artifacts_on_error_to_s3,
            s3_bucket_name=self.s3_bucket_name,
            is_error_log_checked_after_each_step=self.is_error_log_checked_after_each_step,
            is_observation_space_flattened=self.observation_context.is_space_flattened,
            is_godot_restarted_on_error=self.is_godot_restarted_on_error,
            is_dev_flag_added=self.is_dev_flag_added,
            run_uuid=run_uuid,
        )

    def get_replay(self, world_path: Optional[str] = None) -> "GodotEnvReplay":
        from avalon.datagen.godot_env.replay import GodotEnvReplay

        if not self.is_reset_called_already:
            logger.warning(
                f"{self}.get_replay creates a new GodotEnv, but this env is not started. "
                f"Consider using GodotEnvReplay.from_env instead."
            )
        return GodotEnvReplay.from_env(self._spawn_fresh_copy_of_env(self.process.run_uuid), world_path)

    def _get_world_params_by_id(self, world_id: Optional[int]) -> GeneratedWorldParamsType:
        return only(self.world_generator.generate_batch(world_id, 1))


# utility method to get action type given an avalon config. Ideally, this should probably be in a utility class
# but this requires more refactoring due to circular dependency issues


def _randint_of_size(dtype: Type[np.integer]):
    bounds = np.iinfo(dtype)
    return np.random.randint(low=bounds.min, high=bounds.max + 1)
