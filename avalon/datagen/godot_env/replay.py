import json
from pathlib import Path
from typing import Generic
from typing import Iterator
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
from typing import cast

import attr
from numpy import typing as npt

from avalon.common.errors import SwitchError
from avalon.datagen.godot_env.action_log import GodotEnvActionLog
from avalon.datagen.godot_env.actions import ActionType
from avalon.datagen.godot_env.actions import MouseKeyboardAction
from avalon.datagen.godot_env.actions import VRAction
from avalon.datagen.godot_env.goals import GoalProgressResult
from avalon.datagen.godot_env.goals import NullGoalEvaluator
from avalon.datagen.godot_env.godot_env import GodotEnv
from avalon.datagen.godot_env.interactive_godot_process import get_first_run_action_record_path
from avalon.datagen.godot_env.observations import AvalonObservation
from avalon.datagen.godot_env.observations import ObservationType
from avalon.datagen.godot_generated_types import ACTION_MESSAGE
from avalon.datagen.godot_generated_types import DEBUG_CAMERA_ACTION_MESSAGE
from avalon.datagen.godot_generated_types import LOAD_SNAPSHOT_MESSAGE
from avalon.datagen.godot_generated_types import RENDER_MESSAGE
from avalon.datagen.godot_generated_types import RESET_MESSAGE
from avalon.datagen.godot_generated_types import SAVE_SNAPSHOT_MESSAGE
from avalon.datagen.godot_generated_types import SEED_MESSAGE
from avalon.datagen.godot_generated_types import AvalonSimSpec
from avalon.datagen.godot_generated_types import MouseKeyboardAgentPlayerSpec
from avalon.datagen.godot_generated_types import MouseKeyboardHumanPlayerSpec
from avalon.datagen.godot_generated_types import VRAgentPlayerSpec
from avalon.datagen.godot_generated_types import VRHumanPlayerSpec

# Mapping of feature name to (data_type, shape).
from avalon.datagen.world_creation.constants import STARTING_HIT_POINTS
from avalon.datagen.world_creation.world_generator import GeneratedWorldParamsType


@attr.s(auto_attribs=True, frozen=True, slots=True)
class GodotEnvReplay(Generic[ObservationType, ActionType, GeneratedWorldParamsType]):
    env: GodotEnv[ObservationType, ActionType, GeneratedWorldParamsType]
    action_log: GodotEnvActionLog[ActionType]
    world_path: Optional[str] = None

    @classmethod
    def from_local_files(
        cls, run_uuid: str, action_path: Path, config_path: Path, world_path: Path
    ) -> "GodotEnvReplay":
        reconstructed_config = AvalonSimSpec.from_dict(json.load(open(config_path, "r")))
        action_type = get_action_type_from_config(reconstructed_config)

        env = cast(
            GodotEnv[ObservationType, ActionType, GeneratedWorldParamsType],
            GodotEnv(
                config=reconstructed_config,
                observation_type=AvalonObservation,
                action_type=action_type,
                # TODO why do we need the null goal evaluator?
                goal_evaluator=NullGoalEvaluator(),
                run_uuid=run_uuid,
                is_logging_artifacts_on_error_to_s3=False,
            ),
        )

        log = GodotEnvActionLog.parse(str(action_path), env.action_type)
        return cls(env, log, str(world_path))

    @classmethod
    def from_env(
        cls, env: GodotEnv[ObservationType, ActionType, GeneratedWorldParamsType], world_path: Optional[str]
    ) -> "GodotEnvReplay":
        action_record_path = str(get_first_run_action_record_path(env.process.config_path))
        log = GodotEnvActionLog.parse(action_record_path, env.action_type)
        return cls(env, log, world_path)

    def __attrs_post_init__(self) -> None:
        env = self.env
        action_log = self.action_log
        assert not env.is_reset_called_already, f"{env} looks like it has already been run."
        selected_feature_names = list(env.observation_context.selected_features.keys())
        assert selected_feature_names == action_log.selected_features, (
            f"{env}'s selected_features names {selected_feature_names} does not match "
            f"{action_log.selected_features} logged in {action_log.path}"
        )

    def __call__(self) -> Iterator[Union[Tuple[ObservationType, Optional[GoalProgressResult]], npt.NDArray, Path]]:
        env = self.env
        action_log = self.action_log
        world_path = self.world_path

        for message in action_log.messages:
            if message[0] == ACTION_MESSAGE:
                action = message[1]
                yield env.act(action)
            elif message[0] == DEBUG_CAMERA_ACTION_MESSAGE:
                debug_action = message[1]
                yield env.debug_act(debug_action), None
            elif message[0] == RESET_MESSAGE:
                # if we have the world path in the env replayer then we can reset with that specific world
                if world_path is not None:
                    yield env.reset_nicely_with_specific_world(
                        episode_seed=message[2],
                        world_path=world_path,
                        starting_hit_points=STARTING_HIT_POINTS,
                    ), None
                # else, create a new world when resetting
                else:
                    yield env.reset_nicely(episode_seed=message[2]), None

            elif message[0] == RENDER_MESSAGE:
                yield env.render()
            elif message[0] == SEED_MESSAGE:
                episode_id = message[1]
                yield env.seed_nicely(episode_id)
            elif message[0] == SAVE_SNAPSHOT_MESSAGE:
                yield env.save_snapshot()
            elif message[0] == LOAD_SNAPSHOT_MESSAGE:
                snapshot_path = message[1]
                yield env.load_snapshot(Path(snapshot_path))
            else:
                raise SwitchError(f"Invalid replay message {message}")


def get_action_type_from_config(config: AvalonSimSpec) -> Union[Type[MouseKeyboardAction], Type[VRAction]]:
    if isinstance(config.player, (MouseKeyboardAgentPlayerSpec, MouseKeyboardHumanPlayerSpec)):
        return MouseKeyboardAction
    elif isinstance(config.player, (VRAgentPlayerSpec, VRHumanPlayerSpec)):
        return VRAction
    else:
        raise SwitchError(config.player)
