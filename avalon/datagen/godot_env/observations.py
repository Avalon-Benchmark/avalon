from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import Final
from typing import Generic
from typing import Iterable
from typing import List
from typing import Optional
from typing import Protocol
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

import attr
import numpy as np
from gym import spaces
from numpy import typing as npt

from avalon.common.type_utils import assert_not_none
from avalon.common.utils import only
from avalon.datagen.godot_generated_types import FAKE_TYPE_IMAGE as GENERATED_FAKE_TYPE_IMAGE

# Mapping of feature name to (data_type, shape).
FeatureSpecDict = OrderedDict[str, Tuple[int, Tuple[int, ...]]]

# Data dictionary of features, interpretable by a `FeatureSpecDict`
FeatureDataDict = Dict[str, npt.NDArray]


OPTIONAL_FEATURES = ["isometric_rgbd", "top_down_rgbd"]

FAKE_TYPE_IMAGE: Final = GENERATED_FAKE_TYPE_IMAGE
TYPE_INT: Final = 2
TYPE_REAL: Final = 3
TYPE_VECTOR2: Final = 5
TYPE_VECTOR3: Final = 7
TYPE_QUAT: Final = 10

NP_DTYPE_MAP: Final[Dict[int, Type[np.number]]] = {
    TYPE_INT: np.int32,
    TYPE_REAL: np.float32,
    TYPE_VECTOR2: np.float32,
    TYPE_VECTOR3: np.float32,
    TYPE_QUAT: np.float32,
    FAKE_TYPE_IMAGE: np.uint8,
}


class InvalidObservationType(Exception):
    pass


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


class GodotObservationContext(Generic[ObservationType]):
    def __init__(
        self,
        observation_type: Type[ObservationType],
        is_space_flattened: bool,
        available_features: FeatureSpecDict,
    ) -> None:
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

    def _select_features(self) -> FeatureSpecDict:
        # validate our ObservationType and get selected_features
        selected_features: FeatureSpecDict = OrderedDict()
        for field in self.observation_type.get_selected_features():
            type_and_dims = self.available_features.get(field, None)
            # TODO: think of a cleaner way to make these optional?
            if type_and_dims is None and field in OPTIONAL_FEATURES:
                continue
            elif type_and_dims is None and field not in OPTIONAL_FEATURES:
                raise InvalidObservationType(
                    f"Could not find requested feature '{field}'. Available features are: {list(self.available_features)}"
                )
            # TODO: check that the types line up
            assert type_and_dims is not None
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


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class AvalonObservation(AttrsObservation):
    rgbd: npt.NDArray[np.uint8]
    episode_id: npt.NDArray[np.int32]
    frame_id: npt.NDArray[np.int32]
    reward: npt.NDArray[np.float32]
    is_done: npt.NDArray[np.float32]
    is_dead: npt.NDArray[np.float32]

    target_head_position: npt.NDArray[np.float32]
    target_left_hand_position: npt.NDArray[np.float32]
    target_right_hand_position: npt.NDArray[np.float32]

    target_head_rotation: npt.NDArray[np.float32]
    target_left_hand_rotation: npt.NDArray[np.float32]
    target_right_hand_rotation: npt.NDArray[np.float32]

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
    hit_points_lost_from_enemies: npt.NDArray[np.float32]
    hit_points_gained_from_eating: npt.NDArray[np.float32]
    hit_points: npt.NDArray[np.float32]

    # TODO: think of a cleaner way to make these optional?
    isometric_rgbd: npt.NDArray[np.uint8] = np.array([], dtype=np.uint8)
    top_down_rgbd: npt.NDArray[np.uint8] = np.array([], dtype=np.uint8)

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
            "hit_points_lost_from_enemies",
            "hit_points_gained_from_eating",
            "hit_points",
            "frame_id",  # TODO: this will actually be frames_remaining
        )


def _get_default_space(data_type: int, dims: Tuple[int, ...]) -> spaces.Space:
    np_dtype = NP_DTYPE_MAP[data_type]
    if data_type == FAKE_TYPE_IMAGE:
        return spaces.Box(low=0, high=255, shape=dims, dtype=np_dtype)
    return spaces.Box(low=-np.inf, high=np.inf, shape=dims, dtype=np_dtype)


def flatten_observation_space(observation_space: spaces.Dict, flattened_observation_keys: List[str]) -> spaces.Space:
    lows, highs = [], []
    dtypes: Set[np.dtype] = set()
    for key in flattened_observation_keys:
        space = observation_space[key]
        assert isinstance(space, spaces.Box)
        lows.append(np.zeros(assert_not_none(space.shape), dtype=space.dtype) + space.low)
        highs.append(np.zeros(assert_not_none(space.shape), dtype=space.dtype) + space.high)
        dtypes.add(assert_not_none(space.dtype))

    dtype = only(dtypes)
    flat_lows = np.concatenate(lows, axis=-1)
    flat_highs = np.concatenate(highs, axis=-1)
    shape = flat_lows.shape
    return spaces.Box(low=flat_lows, high=flat_highs, shape=shape, dtype=assert_not_none(dtype))  # type: ignore[arg-type]


def flatten_observation(observation: ObservationType, flattened_observation_keys: List[str]) -> np.ndarray:
    return np.concatenate([getattr(observation, x) for x in flattened_observation_keys], axis=-1)
