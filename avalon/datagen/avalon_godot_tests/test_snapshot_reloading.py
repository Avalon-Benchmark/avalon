import uuid
from pathlib import Path
from typing import Callable
from typing import List

from avalon.contrib.testing_utils import fixture
from avalon.contrib.testing_utils import slow_integration_test
from avalon.contrib.testing_utils import use
from avalon.datagen.avalon_godot_tests.conftest import behavior_test_folder_
from avalon.datagen.avalon_godot_tests.conftest import godot_env_
from avalon.datagen.avalon_godot_tests.scenario import AvalonEnv
from avalon.datagen.avalon_godot_tests.scenario import BehaviorManifest
from avalon.datagen.avalon_godot_tests.scenario import Scenario
from avalon.datagen.avalon_godot_tests.snapshots import SnapshotCollection
from avalon.datagen.avalon_godot_tests.snapshots import combined_snapshot_rgbd
from avalon.datagen.avalon_godot_tests.snapshots import observe_snapshots
from avalon.datagen.avalon_godot_tests.snapshots import percent_difference
from avalon.datagen.avalon_godot_tests.snapshots import run_scenarios_with_snapshots
from avalon.datagen.avalon_godot_tests.test_animal_behaviors import climb_behaviors_
from avalon.datagen.avalon_godot_tests.test_animal_behaviors import inactive_behaviors_
from avalon.datagen.avalon_godot_tests.test_animal_behaviors import item_interaction_behaviors_
from avalon.datagen.avalon_godot_tests.test_animal_behaviors import predator_chase_behaviors_
from avalon.datagen.avalon_godot_tests.test_animal_behaviors import prey_flee_behaviors_
from avalon.datagen.avalon_godot_tests.test_item_behaviors import tree_food_behaviors_

_SnapshotObserver = Callable[[List[Scenario]], List[SnapshotCollection]]


@fixture
def snapshot_behavior_manifest_() -> BehaviorManifest:
    return BehaviorManifest.load(Path(__file__).parent / "data/snapshot_behavior_manifest.json")


@fixture
def snapshot_count_() -> int:
    return 10


@fixture
@use(behavior_test_folder_, godot_env_, snapshot_count_)
def observe_all_snapshots_(
    behavior_test_folder: Path,
    godot_env: AvalonEnv,
    snapshot_count: int,
) -> _SnapshotObserver:
    scenario_path = behavior_test_folder / "snapshot_scenarios"

    def observe_all(scenarios: List[Scenario]) -> List[SnapshotCollection]:
        scenarios_with_snapshots = run_scenarios_with_snapshots(godot_env, scenarios, scenario_path, snapshot_count)
        return [
            observe_snapshots(original_observation, lambda: godot_env)
            for original_observation in scenarios_with_snapshots
        ]

    return observe_all


@fixture
@use(observe_all_snapshots_, inactive_behaviors_)
def animal_inactive_snapshot_behaviors_(
    observe_all_snapshots: _SnapshotObserver, inactive_behaviors: List[Scenario]
) -> List[SnapshotCollection]:
    return observe_all_snapshots(inactive_behaviors)


@fixture
@use(observe_all_snapshots_, prey_flee_behaviors_)
def prey_flee_snapshot_behaviors_(
    observe_all_snapshots: _SnapshotObserver, prey_flee_behaviors: List[Scenario]
) -> List[SnapshotCollection]:
    return observe_all_snapshots(prey_flee_behaviors)


@fixture
@use(observe_all_snapshots_, predator_chase_behaviors_)
def predator_chase_snapshot_behaviors_(
    observe_all_snapshots: _SnapshotObserver, predator_chase_behaviors: List[Scenario]
) -> List[SnapshotCollection]:
    return observe_all_snapshots(predator_chase_behaviors)


@fixture
@use(observe_all_snapshots_, climb_behaviors_)
def animal_climb_snapshot_behaviors_(
    observe_all_snapshots: _SnapshotObserver, climb_behaviors: List[Scenario]
) -> List[SnapshotCollection]:
    return observe_all_snapshots(climb_behaviors)


@fixture
@use(observe_all_snapshots_, item_interaction_behaviors_)
def animal_item_interaction_snapshot_behaviors_(
    observe_all_snapshots: _SnapshotObserver,
    item_interaction_behaviors: List[Scenario],
) -> List[SnapshotCollection]:
    return observe_all_snapshots(item_interaction_behaviors)


@fixture
@use(observe_all_snapshots_, tree_food_behaviors_)
def tree_food_snapshot_behaviors_(
    observe_all_snapshots: _SnapshotObserver,
    tree_food_behaviors: List[Scenario],
) -> List[SnapshotCollection]:
    return observe_all_snapshots(tree_food_behaviors)


@fixture
@use(godot_env_, animal_item_interaction_snapshot_behaviors_)
def animal_item_interaction_snapshot_behaviors_in_unique_envs_(
    godot_env: AvalonEnv,
    animal_item_interaction_snapshot_behaviors: List[SnapshotCollection],
) -> List[SnapshotCollection]:
    def get_unique_env():
        return godot_env._spawn_fresh_copy_of_env(str(uuid.uuid4()))

    all_original_observations = [sc.original_observations for sc in animal_item_interaction_snapshot_behaviors]
    return [
        observe_snapshots(original_observations, get_unique_env) for original_observations in all_original_observations
    ]


@fixture
@use(godot_env_, tree_food_snapshot_behaviors_)
def tree_food_snapshot_behaviors_in_unique_envs_(
    godot_env: AvalonEnv,
    tree_food_snapshot_behaviors: List[SnapshotCollection],
) -> List[SnapshotCollection]:
    def get_unique_env():
        return godot_env._spawn_fresh_copy_of_env(str(uuid.uuid4()))

    all_original_observations = [sc.original_observations for sc in tree_food_snapshot_behaviors]
    return [
        observe_snapshots(original_observations, get_unique_env) for original_observations in all_original_observations
    ]


@slow_integration_test
@use(snapshot_behavior_manifest_, animal_inactive_snapshot_behaviors_)
def test_animal_inactive_snapshot_behavior_regressions(
    snapshot_behavior_manifest: BehaviorManifest,
    animal_inactive_snapshot_behaviors: List[SnapshotCollection],
):
    for scenario_snapshots in animal_inactive_snapshot_behaviors:
        verify_combined_snapshot_observations(snapshot_behavior_manifest, scenario_snapshots)


@slow_integration_test
@use(snapshot_behavior_manifest_, prey_flee_snapshot_behaviors_)
def test_prey_flee_snapshot_behavior_regressions(
    snapshot_behavior_manifest: BehaviorManifest,
    prey_flee_snapshot_behaviors: List[SnapshotCollection],
):
    for scenario_snapshots in prey_flee_snapshot_behaviors:
        verify_combined_snapshot_observations(snapshot_behavior_manifest, scenario_snapshots)


@slow_integration_test
@use(snapshot_behavior_manifest_, predator_chase_snapshot_behaviors_)
def test_predator_chase_snapshot_behavior_regressions(
    snapshot_behavior_manifest: BehaviorManifest,
    predator_chase_snapshot_behaviors: List[SnapshotCollection],
):
    for scenario_snapshots in predator_chase_snapshot_behaviors:
        verify_combined_snapshot_observations(snapshot_behavior_manifest, scenario_snapshots)


@slow_integration_test
@use(snapshot_behavior_manifest_, animal_climb_snapshot_behaviors_)
def test_animal_climb_snapshot_behavior_regressions(
    snapshot_behavior_manifest: BehaviorManifest,
    animal_climb_snapshot_behaviors: List[SnapshotCollection],
):
    for scenario_snapshots in animal_climb_snapshot_behaviors:
        verify_combined_snapshot_observations(snapshot_behavior_manifest, scenario_snapshots)


@slow_integration_test
@use(snapshot_behavior_manifest_, animal_item_interaction_snapshot_behaviors_)
def test_animal_item_interaction_snapshot_behavior_regressions(
    snapshot_behavior_manifest: BehaviorManifest,
    animal_item_interaction_snapshot_behaviors: List[SnapshotCollection],
):
    for scenario_snapshots in animal_item_interaction_snapshot_behaviors:
        verify_combined_snapshot_observations(snapshot_behavior_manifest, scenario_snapshots)


@slow_integration_test
@use(snapshot_behavior_manifest_, tree_food_snapshot_behaviors_)
def test_tree_food_snapshot_behavior_regressions(
    snapshot_behavior_manifest: BehaviorManifest,
    tree_food_snapshot_behaviors: List[SnapshotCollection],
):
    for scenario_snapshots in tree_food_snapshot_behaviors:
        verify_combined_snapshot_observations(snapshot_behavior_manifest, scenario_snapshots)


@slow_integration_test
@use(animal_item_interaction_snapshot_behaviors_, animal_item_interaction_snapshot_behaviors_in_unique_envs_)
def test_animal_item_interaction_snapshot_regressions_in_unique_envs(
    animal_item_interaction_snapshot_behaviors: List[SnapshotCollection],
    animal_item_interaction_snapshot_behaviors_in_unique_envs: List[SnapshotCollection],
):
    same_envs_vs_unique = zip(
        animal_item_interaction_snapshot_behaviors, animal_item_interaction_snapshot_behaviors_in_unique_envs
    )
    for same_env, unique_envs in same_envs_vs_unique:
        # TODO this difference should be much lower (0 really)
        acceptable_difference = 0.01
        if same_env.manifest_key == "snake_unharmed_by_held_rock_snapshots":
            acceptable_difference = 0.02  # :(
        verify_low_difference(same_env, unique_envs, acceptable_difference)


@slow_integration_test
@use(tree_food_snapshot_behaviors_, tree_food_snapshot_behaviors_in_unique_envs_)
def test_tree_food_snapshot_regressions_in_unique_envs(
    tree_food_snapshot_behaviors: List[SnapshotCollection],
    tree_food_snapshot_behaviors_in_unique_envs: List[SnapshotCollection],
):
    for same_env, unique_envs in zip(tree_food_snapshot_behaviors, tree_food_snapshot_behaviors_in_unique_envs):
        verify_low_difference(same_env, unique_envs, acceptable_difference=0.005)


def verify_combined_snapshot_observations(manifest: BehaviorManifest, collection: SnapshotCollection):
    historical_checksums = manifest.get_checksums(collection.manifest_key)
    checksums = collection._checksum_dict(is_testing=True)
    assert checksums == historical_checksums, f"{collection.manifest_key}'s checksums have changed!"


def verify_low_difference(a: SnapshotCollection, b: SnapshotCollection, acceptable_difference: float):
    a_rgbd = combined_snapshot_rgbd(a.snapshots)
    b_rgbd = combined_snapshot_rgbd(b.snapshots)
    difference = percent_difference(a_rgbd, b_rgbd)
    assert (
        difference < acceptable_difference
    ), f"{a.manifest_key} run in different envs yielded too many differing pixels ({difference})"
