from typing import Any
from typing import Dict
from typing import Generic
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import TypeVar

import attr

from avalon.common.errors import SwitchError
from avalon.datagen.godot_env.observations import AvalonObservation

# Mapping of feature name to (data_type, shape).
from avalon.datagen.world_creation.constants import AvalonTask
from avalon.datagen.world_creation.constants import avalon_task_to_int
from avalon.datagen.world_creation.world_generator import GenerateAvalonWorldParams
from avalon.datagen.world_creation.world_generator import GeneratedAvalonWorldParams
from avalon.datagen.world_creation.world_generator import GeneratedWorldParamsType

ObsType = TypeVar("ObsType")


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class GoalProgressResult:
    reward: float
    is_done: bool
    log: Dict[str, Any]
    world_path: Optional[str] = None


class GoalEvaluator(Generic[ObsType, GeneratedWorldParamsType]):
    def calculate_next_is_done_and_reward(self, observation: ObsType) -> Tuple[bool, float]:
        result = self.calculate_goal_progress(observation)
        return result.is_done, result.reward

    def calculate_goal_progress(self, observation: ObsType) -> GoalProgressResult:
        raise NotImplementedError()

    def reset(self, observation: ObsType, world_params: Optional[GeneratedWorldParamsType] = None) -> None:
        raise NotImplementedError()


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class NullGoalEvaluator(GoalEvaluator[AvalonObservation, GeneratedAvalonWorldParams]):
    def calculate_goal_progress(self, observation: AvalonObservation) -> GoalProgressResult:
        return GoalProgressResult(reward=0, is_done=False, log={})

    def reset(self, observation: AvalonObservation, world_params: Optional[GeneratedAvalonWorldParams] = None) -> None:
        pass


FRAMES_PER_MINUTE = 600


@attr.s(auto_attribs=True, collect_by_mro=True)
class AvalonGoalEvaluator(GoalEvaluator[AvalonObservation, GeneratedAvalonWorldParams]):
    score_cost_per_frame: float = 1e-4
    frame_limit: int = 9000

    def calculate_goal_progress(self, observation: AvalonObservation) -> GoalProgressResult:
        remaining_frames = self.frame_limit - observation.frame_id.item() - 1
        if not self.is_all_food_eaten:  # type: ignore[has-type]
            self.score = max(
                0,
                observation.hit_points.item() + remaining_frames * self.score_cost_per_frame - self.initial_hit_points,
            )
        self.is_all_food_eaten = bool(observation.is_food_present_in_world.item() < 0.5)

        is_done = observation.is_done.item() > 0

        truncated = False
        if not is_done and remaining_frames == 0:
            is_done = True
            truncated = True

        assert self.world_params is not None
        return GoalProgressResult(
            is_done=is_done,
            reward=self.get_reward(observation),
            log={
                "success": 1 if self.is_all_food_eaten else 0,
                "score": self.score,
                "remaining_frames": remaining_frames,
                "frame_id": observation.frame_id.item(),
                "difficulty": self.world_params.difficulty,
                # make this an int to work with our rollout worker
                "task": avalon_task_to_int[self.world_params.task.value],
                "episode_id": observation.episode_id.item(),
                "world_index": self.world_params.index,
                "hit_points": observation.hit_points.item(),
                "TimeLimit.truncated": truncated,
            },
            world_path=self.world_params.output,
        )

    def get_reward(self, observation: AvalonObservation) -> float:
        return observation.reward.item() - self.score_cost_per_frame

    def get_level_frame_limit(self, world_params: GenerateAvalonWorldParams) -> int:
        if world_params.task in (AvalonTask.FIND, AvalonTask.GATHER, AvalonTask.NAVIGATE):
            return 15 * FRAMES_PER_MINUTE
        elif world_params.task in (AvalonTask.SURVIVE, AvalonTask.STACK, AvalonTask.CARRY, AvalonTask.EXPLORE):
            return 10 * FRAMES_PER_MINUTE
        else:
            return 5 * FRAMES_PER_MINUTE

    def reset(self, observation: AvalonObservation, world_params: Optional[GenerateAvalonWorldParams] = None) -> None:
        assert world_params is not None
        self.world_params = world_params
        self.initial_hit_points = observation.hit_points.item()
        self.frame_limit = self.get_level_frame_limit(world_params)
        self.score = self.frame_limit * self.score_cost_per_frame
        self.is_all_food_eaten = bool(observation.is_food_present_in_world.item() < 0.5)


# These are calculated such that 98% of humans succeed after applying the dynamic frame limit
TRAINING_FRAME_LIMITS_AT_ZERO_DIFFICULTY = {
    AvalonTask.EAT: 260,
    AvalonTask.FIGHT: 220,
    AvalonTask.PUSH: 700,
    AvalonTask.DESCEND: 340,
    AvalonTask.STACK: 710,
    AvalonTask.CLIMB: 250,
    AvalonTask.SCRAMBLE: 190,
    AvalonTask.THROW: 600,
    AvalonTask.GATHER: 3300,
    AvalonTask.CARRY: 610,
    AvalonTask.HUNT: 420,
    AvalonTask.FIND: 1180,
    AvalonTask.EXPLORE: 1340,
    AvalonTask.MOVE: 360,
    AvalonTask.BRIDGE: 630,
    AvalonTask.SURVIVE: 3780,
    AvalonTask.OPEN: 170,
    AvalonTask.NAVIGATE: 1220,
    AvalonTask.AVOID: 240,
    AvalonTask.JUMP: 310,
}


@attr.s(auto_attribs=True, collect_by_mro=True)
class TrainingAvalonGoalEvaluator(AvalonGoalEvaluator):
    energy_cost_coefficient: float = 1e-9
    body_ke_coefficient: float = 0.0
    body_pe_coefficient: float = 1.0
    head_pe_coefficient: float = 1.0
    hand_ke_coefficient: float = 0.0
    hand_pe_coefficient: float = 1.0
    head_roll_coefficient: float = 0.0
    head_pitch_coefficient: float = 0.0
    energy_cost_aggregator: Literal["sum", "max"] = "sum"

    def get_level_frame_limit(self, world_params: GenerateAvalonWorldParams) -> int:
        super_frame_limit = super().get_level_frame_limit(world_params)
        dynamic_frame_limit = int(
            TRAINING_FRAME_LIMITS_AT_ZERO_DIFFICULTY[world_params.task] * 10**world_params.difficulty
        )
        return min(super_frame_limit, dynamic_frame_limit)

    def get_reward(self, observation: AvalonObservation) -> float:
        energy_cost = (
            self.energy_cost_coefficient
            * (
                self.head_pe_coefficient * observation.physical_head_potential_energy_expenditure.item()
                + self.body_ke_coefficient * observation.physical_body_kinetic_energy_expenditure.item()
                + self.body_pe_coefficient * observation.physical_body_potential_energy_expenditure.item()
                + self.hand_ke_coefficient * observation.physical_left_hand_kinetic_energy_expenditure.item()
                + self.hand_pe_coefficient * observation.physical_left_hand_potential_energy_expenditure.item()
                + self.hand_ke_coefficient * observation.physical_right_hand_kinetic_energy_expenditure.item()
                + self.hand_pe_coefficient * observation.physical_right_hand_potential_energy_expenditure.item()
            )
            + self.head_roll_coefficient * observation.physical_head_rotation[2].item()
            + self.head_pitch_coefficient * max(abs(observation.physical_head_rotation[0].item()) - 45.0, 0.0)
        )
        if self.energy_cost_aggregator == "sum":
            energy_cost += self.score_cost_per_frame
        elif self.energy_cost_aggregator == "max":
            energy_cost = max(self.score_cost_per_frame, energy_cost)
        else:
            raise SwitchError(f"Invalid energy_cost_aggregator: {self.energy_cost_aggregator}")
        return float(observation.reward.item() - energy_cost)
