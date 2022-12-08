from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation

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
from avalon.datagen.godot_env.actions import DebugCameraAction
from avalon.datagen.godot_env.actions import VRAction
from avalon.datagen.world_creation.entities.animals import ALL_PREDATOR_CLASSES
from avalon.datagen.world_creation.entities.animals import ALL_PREY_CLASSES
from avalon.datagen.world_creation.entities.animals import Alligator
from avalon.datagen.world_creation.entities.animals import Animal
from avalon.datagen.world_creation.entities.animals import Bear
from avalon.datagen.world_creation.entities.animals import Bee
from avalon.datagen.world_creation.entities.animals import Crow
from avalon.datagen.world_creation.entities.animals import Deer
from avalon.datagen.world_creation.entities.animals import Eagle
from avalon.datagen.world_creation.entities.animals import Hawk
from avalon.datagen.world_creation.entities.animals import Hippo
from avalon.datagen.world_creation.entities.animals import Jaguar
from avalon.datagen.world_creation.entities.animals import Mouse
from avalon.datagen.world_creation.entities.animals import Pigeon
from avalon.datagen.world_creation.entities.animals import Rabbit
from avalon.datagen.world_creation.entities.animals import Snake
from avalon.datagen.world_creation.entities.animals import Squirrel
from avalon.datagen.world_creation.entities.animals import Turtle
from avalon.datagen.world_creation.entities.animals import Wolf
from avalon.datagen.world_creation.entities.constants import FOOD_TREE_BASE_HEIGHT
from avalon.datagen.world_creation.entities.food import FoodTree
from avalon.datagen.world_creation.entities.item import Item
from avalon.datagen.world_creation.entities.pillar import Pillar
from avalon.datagen.world_creation.entities.tools.weapons import LargeRock
from avalon.datagen.world_creation.entities.tools.weapons import LargeStick
from avalon.datagen.world_creation.entities.tools.weapons import Stick
from avalon.datagen.world_creation.types import Point3DNP

_ScenarioActions = Union[DebugCameraAction, VRAction]


@fixture
def animal_behavior_manifest_() -> BehaviorManifest:
    return BehaviorManifest.load(Path(__file__).parent / "data/animal_behavior_manifest.json")


@fixture
def inactive_behaviors_() -> List[Scenario]:
    scenarios = []
    all_animals: List[Type[Animal]] = [*ALL_PREY_CLASSES, *ALL_PREDATOR_CLASSES]
    periodic = (Turtle, Jaguar, Hawk)
    static = (Snake, Hippo, Deer)
    for animal_type in all_animals:
        frames = 200
        if animal_type in periodic:
            frames = 400
        elif animal_type in static:
            frames = 50
        scenarios.append(Scenario.inactive(animal_type, frames))
    return scenarios


@fixture
def prey_flee_behaviors_() -> List[Scenario]:
    near = 7.0
    far = 13.0
    prey_defaults: Dict[str, Any] = dict(
        stop_frames=0,
        camera_distance=-6.0,
        is_facing_player=True,
        is_zig_zagging=False,
    )
    one_big_zig_zag: Dict[str, Any] = dict(
        run_frames=50,
        final_actions=[
            *_turn_and_run(-1.0, 25, 2, 90),
            *_turn_and_run(-1.0, 25, 4, -90),
            *_turn_and_run(-1.0, 100, 2, 90),
        ],
    )
    return [
        Scenario.pursuit("flee", Turtle, player_speed=-0.5, spawn_distance=near, **one_big_zig_zag, **prey_defaults),
        Scenario.pursuit("flee", Mouse, player_speed=-1.0, spawn_distance=6.0, **one_big_zig_zag, **prey_defaults),
        Scenario.pursuit(
            "flee",
            Rabbit,
            player_speed=-1.0,
            spawn_distance=far,
            **prey_defaults,
            run_frames=50,
            final_actions=[
                *_turn_and_run(-1.0, 150, 2, 90),
            ],
        ),
        Scenario.pursuit("flee", Pigeon, player_speed=-1.0, spawn_distance=near, return_frame=100, **prey_defaults),
        Scenario.pursuit("flee", Squirrel, player_speed=-1.0, spawn_distance=near, **one_big_zig_zag, **prey_defaults),
        Scenario.pursuit("flee", Crow, player_speed=-1.0, spawn_distance=near, **prey_defaults),
        Scenario.pursuit("flee", Deer, player_speed=-1.0, spawn_distance=far, **one_big_zig_zag, **prey_defaults),
    ]


@fixture
def predator_chase_behaviors_() -> List[Scenario]:
    return [
        Scenario.pursuit("chase", Bee, player_speed=1.0, spawn_distance=4.0, camera_distance=6.0),
        Scenario.pursuit(
            "avoid_then_chase", Snake, player_speed=-0.5, return_frame=75, return_multiplier=3, is_zig_zagging=False
        ),
        Scenario.pursuit("chase", Hawk, player_speed=1.5, return_frame=40, run_frames=120, stop_frames=130),
        Scenario.pursuit(
            "avoid_then_chase",
            Hippo,
            player_speed=-0.625,
            return_frame=50,
            return_multiplier=1.75,
            is_zig_zagging=False,
        ),
        Scenario.pursuit("chase", Alligator, player_speed=0.75),
        Scenario.pursuit(
            "chase",
            Eagle,
            wait_frames=10,
            player_speed=1.6,
            is_player_facing_away=True,
            run_frames=175,
            stop_frames=25,
            final_actions=_turn_around_and_backup(1.0, 50),
        ),
        Scenario.pursuit("chase", Wolf, player_speed=1.0),
        Scenario.pursuit("chase", Jaguar, player_speed=1.5, is_facing_player=True),
        Scenario.pursuit("chase", Bear, player_speed=1.5),
    ]


@fixture
def climb_behaviors_() -> List[Scenario]:
    def climb_scene(
        animal_type: Type[Animal],
        actions: List[VRAction],
        behavior: str,
        detection_radius_override: Optional[int] = None,
    ):
        name = animal_type.__name__.lower()
        return Scenario(
            f"{name}_climbs_{behavior}",
            animal_type,
            [Scenario.look_at(name, 20.0 if name == "jaguar" else 8.0)] + actions,
            size_in_meters=200,
            animal_position=np.array([-9.0, 0.0, -3.5]),
            spawn_point=(-6.0, -10.0) if animal_type in (Bear, Jaguar) else (-8.0, -2.5),
            cliff_height=10.0,
            is_animal_facing_player=True,
            detection_radius_override=detection_radius_override,
        )

    null_actions = [get_vr_action()]
    slow_backup = [get_vr_action(head_x=-0.25, head_z=-0.25)]
    fast_backup = [get_vr_action(head_x=-1.5, head_z=-1.5)]
    fidget_8_steps = 4 * [get_vr_action(head_x=-1.5, head_z=-1.5)] + 4 * [get_vr_action(head_x=1.5, head_z=1.5)]
    slow_chase = [get_vr_action(head_x=-0.25)]
    fast_retreat = [get_vr_action(head_x=1.5)]

    return [
        climb_scene(Bear, 79 * null_actions + 70 * slow_backup, "in_pursuit"),
        climb_scene(Jaguar, 10 * fidget_8_steps + 69 * slow_backup, "in_pursuit", detection_radius_override=24),
        climb_scene(Squirrel, 10 * null_actions + 69 * slow_chase + 70 * null_actions, "in_flight"),
        climb_scene(Mouse, 10 * null_actions + 69 * slow_chase + 70 * null_actions, "in_flight"),
        climb_scene(Bear, 79 * null_actions + 120 * fast_backup, "down_when_deactivated"),
        climb_scene(
            Jaguar, 10 * fidget_8_steps + 119 * null_actions, "down_when_deactivated", detection_radius_override=24
        ),
        climb_scene(Squirrel, 10 * null_actions + 69 * slow_chase + 130 * fast_retreat, "down_when_deactivated"),
        climb_scene(Mouse, 10 * null_actions + 69 * slow_chase + 130 * fast_retreat, "down_when_deactivated"),
    ]


@fixture
def complex_behaviors_() -> List[Scenario]:
    null_actions = [get_vr_action()]
    obstacle_tree = FoodTree(
        scale=(FOOD_TREE_BASE_HEIGHT / FOOD_TREE_BASE_HEIGHT) * np.array([1.0, 1.0, 1.0]),
        # TODO slight differences in tree position can cause rebounds that result in stuck animals
        # position=np.array([-19.5, 0.0, -30.0]),
        position=np.array([-20.5, 0.0, -31.0]),
        is_food_on_tree=False,
    )
    return [
        Scenario(
            "bear_sidesteps_trees",
            Bear,
            # NOTE: entity_ids are now resolved in export, and FoodTrees always come first
            [Scenario.look_at("bear__2", 8.0)] + 99 * null_actions,
            items=[obstacle_tree],
            is_animal_facing_player=True,
            detection_radius_override=100.0,
            animal_position=np.array([-20, 0.0, -45.0]),
            foliage_density_modifier=2.0,
            spawn_point=(-19.5, -18.0),
        ),
        Scenario(
            "hawk_flies_over_pillars",
            Hawk,
            [Scenario.look_at("hawk", 8.0)] + 149 * null_actions,
            items=[
                Pillar(size=np.array([2.5, 15.0, 2.5]), position=np.array([4.0, 0.0, 0.0])),
                Pillar(size=np.array([2.5, 15.0, 2.5]), position=np.array([0.0, 0.0, 4.0])),
            ],
            is_animal_facing_player=True,
            foliage_density_modifier=2.0,
            spawn_point=(5.0, 5.0),
        ),
        # TODO {walker, flier, climber} avoids ocean
        # TODO expand to Jaguar, Mouse, Squirrel, climb deactivation, getting unstuck
    ]


@fixture
def item_interaction_behaviors_() -> List[Scenario]:
    null_actions = (100 - 1) * [get_vr_action()]

    def snake_interaction(
        scenario: str,
        is_recorded: bool,
        items: List[Item],
        actions: List[VRAction] = [],
        animal_position: Point3DNP = np.zeros(3),
    ) -> Scenario:
        if is_recorded:
            actions = list(read_human_recorded_actions(scenario))
        if len(actions) == 0:
            actions = null_actions
        return Scenario(
            scenario,
            Snake,
            actions=[Scenario.look_at("snake", 6.0), *actions],
            detection_radius_override=0.0,
            spawn_point=(0.0, -10) if is_recorded else None,
            is_spawn_hidden=not is_recorded,
            items=items,
            animal_position=animal_position,
        )

    return [
        snake_interaction(
            "snake_killed_by_falling_rock",
            is_recorded=False,
            items=[LargeRock(position=np.array([0.0, 25.0, 0.0]))],
        ),
        snake_interaction(
            "snake_unharmed_by_falling_stick",
            is_recorded=False,
            # TODO unrotated, the stick will fall through the snake and ground.
            # at 45 degrees, they interact but then the stick tumbles through the ground
            # without the offset, the snake gets pushed into the ground
            # in this, we see the stick seem to glance off the snake.
            # will need to assert in the debug log that the collision occurs at a sufficient speed
            items=[
                LargeStick(
                    position=np.array([0.5, 25.0, 0.5]),
                    rotation=Rotation.from_euler("z", 90, degrees=True),
                )
            ],
        ),
        snake_interaction(
            "snake_killed_with_stick",
            is_recorded=True,
            items=[LargeStick(position=np.array([0, 0.35, -5]))],
        ),
        snake_interaction(
            "snake_killed_with_rock",
            is_recorded=True,
            items=[LargeRock(position=np.array([0, 0.35, -5]))],
            animal_position=np.array([1.5, 0.0, -1.0]),
        ),
        snake_interaction(
            "snake_unharmed_by_thrown_stick",
            is_recorded=True,
            items=[Stick(position=np.array([0, 0.35, -5]))],
        ),
        snake_interaction(
            "snake_unharmed_by_held_rock",
            is_recorded=True,
            items=[LargeRock(position=np.array([0, 0.35, -5]))],
        ),
    ]


# TODO: here we copy the *_observation fixture for each scenario list, and then the
# verification test, so that we have some granularity in test output.
# This is annoying, but I haven't yet found a good way to parameterize test fixtures themselves
@fixture
@use(
    behavior_test_folder_,
    godot_env_,
    inactive_behaviors_,
)
def inactive_behavior_observations_(
    behavior_test_folder: Path,
    godot_env: AvalonEnv,
    inactive_behaviors: List[Scenario],
) -> List[ScenarioObservations]:
    scenario_path = behavior_test_folder / "scenarios"
    return [scenario.run(godot_env, scenario_path) for scenario in inactive_behaviors]


@fixture
@use(
    behavior_test_folder_,
    godot_env_,
    prey_flee_behaviors_,
)
def prey_flee_observations_(
    behavior_test_folder: Path,
    godot_env: AvalonEnv,
    prey_flee_behaviors: List[Scenario],
) -> List[ScenarioObservations]:
    scenario_path = behavior_test_folder / "scenarios"
    return [scenario.run(godot_env, scenario_path) for scenario in prey_flee_behaviors]


@fixture
@use(
    behavior_test_folder_,
    godot_env_,
    predator_chase_behaviors_,
)
def predator_chase_observations_(
    behavior_test_folder: Path,
    godot_env: AvalonEnv,
    predator_chase_behaviors: List[Scenario],
) -> List[ScenarioObservations]:
    scenario_path = behavior_test_folder / "scenarios"
    return [scenario.run(godot_env, scenario_path) for scenario in predator_chase_behaviors]


@fixture
@use(
    behavior_test_folder_,
    godot_env_,
    climb_behaviors_,
)
def climb_observations_(
    behavior_test_folder: Path,
    godot_env: AvalonEnv,
    climb_behaviors: List[Scenario],
) -> List[ScenarioObservations]:
    scenario_path = behavior_test_folder / "scenarios"
    return [scenario.run(godot_env, scenario_path) for scenario in climb_behaviors]


@fixture
@use(
    behavior_test_folder_,
    godot_env_,
    complex_behaviors_,
)
def complex_observations_(
    behavior_test_folder: Path,
    godot_env: AvalonEnv,
    complex_behaviors: List[Scenario],
) -> List[ScenarioObservations]:
    scenario_path = behavior_test_folder / "scenarios"
    return [scenario.run(godot_env, scenario_path) for scenario in complex_behaviors]


@fixture
@use(behavior_test_folder_, godot_env_, item_interaction_behaviors_)
def item_interaction_observations_(
    behavior_test_folder: Path,
    godot_env: AvalonEnv,
    item_interaction_behaviors: List[Scenario],
) -> List[ScenarioObservations]:
    scenario_path = behavior_test_folder / "scenarios"
    return [scenario.run(godot_env, scenario_path) for scenario in item_interaction_behaviors]


@slow_integration_test
@use(
    animal_behavior_manifest_,
    inactive_behavior_observations_,
)
def test_inactive_scenario_regressions(
    animal_behavior_manifest: BehaviorManifest,
    inactive_behavior_observations: List[ScenarioObservations],
) -> None:
    verify_observations(animal_behavior_manifest, inactive_behavior_observations)


@slow_integration_test
@use(
    animal_behavior_manifest_,
    prey_flee_observations_,
)
def test_prey_flee_scenario_regressions(
    animal_behavior_manifest: BehaviorManifest,
    prey_flee_observations: List[ScenarioObservations],
) -> None:
    verify_observations(animal_behavior_manifest, prey_flee_observations)


@slow_integration_test
@use(
    animal_behavior_manifest_,
    predator_chase_observations_,
)
def test_predator_chase_scenario_regressions(
    animal_behavior_manifest: BehaviorManifest,
    predator_chase_observations: List[ScenarioObservations],
) -> None:
    verify_observations(animal_behavior_manifest, predator_chase_observations)


@slow_integration_test
@use(
    animal_behavior_manifest_,
    climb_observations_,
)
def test_climb_scenario_regressions(
    animal_behavior_manifest: BehaviorManifest,
    climb_observations: List[ScenarioObservations],
) -> None:
    verify_observations(animal_behavior_manifest, climb_observations)


@slow_integration_test
@use(
    animal_behavior_manifest_,
    complex_observations_,
)
def test_complex_scenario_regressions(
    animal_behavior_manifest: BehaviorManifest,
    complex_observations: List[ScenarioObservations],
) -> None:
    verify_observations(animal_behavior_manifest, complex_observations)


@slow_integration_test
@use(
    animal_behavior_manifest_,
    item_interaction_observations_,
)
def test_item_interaction_scenario_regressions(
    animal_behavior_manifest: BehaviorManifest,
    item_interaction_observations: List[ScenarioObservations],
) -> None:
    verify_observations(animal_behavior_manifest, item_interaction_observations)


def _turn_around_and_backup(player_speed: float, frames: int) -> List[_ScenarioActions]:
    # this isn't right mathmatically but it gets what we want for some reason
    turn_around = 5 * [get_vr_action(head_yaw=np.radians(180))]
    backup = (frames - 5) * [get_vr_action(head_z=player_speed)]
    return [*turn_around, *backup]


def _turn_and_run(player_speed: float, frames: int, turn_frames: int = 2, degrees: int = 90) -> List[_ScenarioActions]:
    turn = turn_frames * [get_vr_action(head_yaw=np.radians(degrees))]
    run = (frames - turn_frames) * [get_vr_action(head_z=player_speed)]
    return [*turn, *run]


# TODO add damage and behavior to debug, easy debug parsing
# TODO Fall damage, sidestep, ocean avoid, climbing
# TODO items: damages animal, doesn't damage when shouldn't, comes to rest
# TODO consolidate manfiest & bucket upload stuff

# TODO possible optimizations:
# * mechanism for scenario order invariance in the same env (allows for sharing env and more granular tests)
# * lazy parameterizer to get best-of-both-worlds with catalog fixtures
# * custom parametrization schemes https://docs.pytest.org/en/7.1.x/how-to/parametrize.html


def verify_observations(manifest: BehaviorManifest, observations: List[ScenarioObservations]) -> None:
    for obs in observations:
        name = obs.scenario.name
        historical_checksums = manifest.get_checksums(name)

        assert obs._checksum_dict == historical_checksums, f"{name}'s checksums have changed! "
