from pathlib import Path
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import attr
import numpy as np

from avalon.common.utils import flatten
from avalon.contrib.testing_utils import fixture
from avalon.contrib.testing_utils import slow_integration_test
from avalon.contrib.testing_utils import use
from avalon.datagen.avalon_godot_tests.conftest import behavior_test_folder_
from avalon.datagen.avalon_godot_tests.conftest import godot_env_
from avalon.datagen.avalon_godot_tests.scenario import AvalonEnv
from avalon.datagen.avalon_godot_tests.scenario import BehaviorManifest
from avalon.datagen.avalon_godot_tests.scenario import Scenario
from avalon.datagen.avalon_godot_tests.scenario import ScenarioObservations
from avalon.datagen.avalon_godot_tests.scenario import get_vr_action
from avalon.datagen.avalon_godot_tests.scenario import read_human_recorded_actions
from avalon.datagen.avalon_godot_tests.test_animal_behaviors import verify_observations
from avalon.datagen.godot_env.actions import DebugCameraAction
from avalon.datagen.world_creation.entities.altar import Altar
from avalon.datagen.world_creation.entities.constants import FOOD_TREE_BASE_HEIGHT
from avalon.datagen.world_creation.entities.food import FOODS as _FOODS
from avalon.datagen.world_creation.entities.food import Avocado
from avalon.datagen.world_creation.entities.food import Banana
from avalon.datagen.world_creation.entities.food import Carrot
from avalon.datagen.world_creation.entities.food import Fig
from avalon.datagen.world_creation.entities.food import Food
from avalon.datagen.world_creation.entities.food import FoodTree
from avalon.datagen.world_creation.entities.food import Honeycomb
from avalon.datagen.world_creation.entities.food import Mulberry
from avalon.datagen.world_creation.entities.item import Item
from avalon.datagen.world_creation.entities.tools.weapons import LargeRock
from avalon.datagen.world_creation.entities.tools.weapons import LargeStick

# TODO(mjr) pylance can't tell they're all foods :eyeroll:
FOODS: Tuple[Food, ...] = _FOODS
GROUND_FOODS: List[Food] = [f for f in FOODS if f.is_found_on_ground]
TREE_FOODS: List[Food] = [f for f in FOODS if f.is_grown_on_trees]

PLAYER_TOOLS: List[Item] = [
    LargeRock(position=np.array([2.0, 0.0, -5.0])),
    LargeStick(position=np.array([-2.0, 0.0, -5.0])),
]


@fixture
def item_behavior_manifest_() -> BehaviorManifest:
    return BehaviorManifest.load(Path(__file__).parent / "data/item_behavior_manifest.json")


@fixture
def falling_food_behaviors_() -> List[Scenario]:
    # NOTE specific heights will result in falling through altar or rock and maybe floor for most foods
    #      17.0, 21.0, 25.0 and up
    fall_pos = np.array([0.0, 20.0, 0.0])
    falling_scenes = flatten(
        [
            (
                food_scenario(attr.evolve(food, position=fall_pos), "fall_on_ground", is_recorded=False),
                food_scenario(
                    attr.evolve(food, position=fall_pos),
                    "fall_on_rock",
                    is_recorded=False,
                    items=[
                        LargeRock(position=np.array([0.0, 0.0, 0.0])),
                    ],
                ),
                food_scenario(
                    attr.evolve(food, position=np.array([0.0, 0.0, 0.0])),
                    "hit_with_rock",
                    is_recorded=False,
                    items=[
                        LargeRock(position=fall_pos),
                    ],
                ),
            )
            for food in FOODS
            if not isinstance(food, Carrot)
        ]
    )
    avocado_doesnt_crack_on_altar = food_scenario(
        Avocado(position=fall_pos),
        "fall_on_altar",
        is_recorded=False,
        items=[Altar.build(2.0)],
    )
    falling_scenes.append(avocado_doesnt_crack_on_altar)
    return falling_scenes


@fixture
def ground_food_behaviors_() -> List[Scenario]:
    grounded = [food_scenario(food, "grounded", is_recorded=True) for food in GROUND_FOODS]
    grounded.append(food_scenario(Fig(), "goes_splat", is_recorded=True))
    return grounded


@fixture
def tree_food_behaviors_() -> List[Scenario]:
    tree = FoodTree.build(tree_height=FOOD_TREE_BASE_HEIGHT, is_food_on_tree=True)
    return [
        food_scenario((tree, Banana()), "on_tree", is_recorded=True),
        food_scenario((tree, Mulberry()), "on_tree", is_recorded=True),
        food_scenario((tree, Honeycomb()), "on_tree", is_recorded=True),
        food_scenario((tree, Honeycomb()), "gets_dirty", is_recorded=True),
    ]


@fixture
@use(
    behavior_test_folder_,
    godot_env_,
    falling_food_behaviors_,
)
def falling_food_observations_(
    behavior_test_folder: Path,
    godot_env: AvalonEnv,
    falling_food_behaviors: List[Scenario],
) -> List[ScenarioObservations]:
    scenario_path = behavior_test_folder / "scenarios"
    return [scenario.run(godot_env, scenario_path) for scenario in falling_food_behaviors]


@fixture
@use(
    behavior_test_folder_,
    godot_env_,
    ground_food_behaviors_,
)
def ground_food_observations_(
    behavior_test_folder: Path,
    godot_env: AvalonEnv,
    ground_food_behaviors: List[Scenario],
) -> List[ScenarioObservations]:
    scenario_path = behavior_test_folder / "scenarios"
    # TODO: Cherry interaction has become slightly off due to minute player changes (doesn't get picked up and eaten).
    #       But is not worth re-recording unless others also get broken.
    return [scenario.run(godot_env, scenario_path) for scenario in ground_food_behaviors]


@fixture
@use(
    behavior_test_folder_,
    godot_env_,
    tree_food_behaviors_,
)
def tree_food_observations_(
    behavior_test_folder: Path,
    godot_env: AvalonEnv,
    tree_food_behaviors: List[Scenario],
) -> List[ScenarioObservations]:
    scenario_path = behavior_test_folder / "scenarios"
    return [scenario.run(godot_env, scenario_path) for scenario in tree_food_behaviors]


@slow_integration_test
@use(
    item_behavior_manifest_,
    falling_food_observations_,
)
def test_falling_food_scenario_regressions(
    item_behavior_manifest: BehaviorManifest,
    falling_food_observations: List[ScenarioObservations],
):
    verify_observations(item_behavior_manifest, falling_food_observations)


@slow_integration_test
@use(
    item_behavior_manifest_,
    ground_food_observations_,
)
def test_ground_food_scenario_regressions(
    item_behavior_manifest: BehaviorManifest,
    ground_food_observations: List[ScenarioObservations],
):
    verify_observations(item_behavior_manifest, ground_food_observations)


@slow_integration_test
@use(
    item_behavior_manifest_,
    tree_food_observations_,
)
def test_tree_food_scenario_regressions(
    item_behavior_manifest: BehaviorManifest,
    tree_food_observations: List[ScenarioObservations],
):
    verify_observations(item_behavior_manifest, tree_food_observations)


def food_scenario(
    food: Union[Food, Tuple[FoodTree, Food]],
    scenario_name: str,
    items: Optional[Sequence[Item]] = None,
    is_recorded: bool = False,
    frames: int = 50,
) -> Scenario:
    if is_on_tree := isinstance(food, tuple):
        actual_food = food[1]
    else:
        actual_food = food
    food_name = actual_food.__class__.__name__.lower()
    scenario = f"{food_name}_{scenario_name}"

    actions = []
    if is_recorded:
        actions = list(read_human_recorded_actions(scenario))
    if len(actions) == 0:
        actions = (frames - 1) * [get_vr_action()]
    if items is None:
        items = PLAYER_TOOLS if is_recorded else []
    look = DebugCameraAction.level(f"{food_name}__2", distance=3.0) if is_on_tree else Scenario.look_at(food_name, 3.0)
    return Scenario(
        scenario,
        None,
        actions=[look, *actions],
        spawn_point=(0.0, -10),
        is_spawn_hidden=False,
        items=[food, *items],
    )
