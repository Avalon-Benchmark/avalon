import hashlib
import math
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import cast

import attr
import numpy as np
import torch
from numpy.typing import NDArray

from avalon.common.utils import flatten
from avalon.datagen.avalon_godot_tests.conftest import AvalonEnv
from avalon.datagen.avalon_godot_tests.scenario import ManifestValue
from avalon.datagen.avalon_godot_tests.scenario import Scenario
from avalon.datagen.avalon_godot_tests.scenario import ScenarioActions
from avalon.datagen.avalon_godot_tests.scenario import ScenarioObservations
from avalon.datagen.avalon_godot_tests.scenario import SnapshotContext
from avalon.datagen.avalon_godot_tests.scenario import np_checksum
from avalon.datagen.avalon_godot_tests.scenario import rgbd_observations
from avalon.datagen.env_helper import observation_video_array
from avalon.datagen.env_helper import rgbd_to_video_array
from avalon.datagen.godot_env.actions import DebugCameraAction
from avalon.datagen.godot_env.actions import VRActionType
from avalon.datagen.godot_env.observations import AvalonObservationType
from avalon.datagen.godot_generated_types import LOAD_SNAPSHOT_MESSAGE
from avalon.datagen.godot_generated_types import SAVE_SNAPSHOT_MESSAGE


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class SnapshotObservations:
    original: ScenarioObservations
    context: SnapshotContext
    index: int
    end_frame: Optional[int]
    observations: List[AvalonObservationType]

    @property
    def dir(self) -> Path:
        return self.context[0]

    @property
    def start_frame(self) -> int:
        return self.context[1]

    @property
    def action_observations(self):
        return self.observations[1:]

    @property
    def expected_observations(self) -> List[AvalonObservationType]:
        # TODO not sure why end_frame-1 is necessary
        return self.original.observations[
            self.start_frame - 1 : self.end_frame - 1 if self.end_frame is not None else None
        ]


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class SnapshotCollection:

    snapshots: List[SnapshotObservations]

    @property
    def combined_video(self):
        return combined_snapshot_videos(self.snapshots)

    @property
    def original_observations(self) -> ScenarioObservations:
        return self.snapshots[0].original

    @property
    def manifest_key(self) -> str:
        scenario_name = self.snapshots[0].original.scenario.name
        return f"{scenario_name}_snapshots"

    def _checksum_dict(self, is_testing: bool = False) -> Dict[str, ManifestValue]:
        return summarize_scenario_snapshots(self.snapshots, is_testing)

    @property
    def checksum_summary(self) -> Dict[str, Dict[str, ManifestValue]]:
        return {self.manifest_key: self._checksum_dict()}

    def is_unchanged_from_version_in(self, checksums: Dict[str, Dict[str, ManifestValue]]) -> bool:
        historical_checksums = checksums.get(self.manifest_key, None)
        return self._checksum_dict() == historical_checksums

    def get_comparable_videos(self, is_prelude_included: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        original = original_rgbd(self.snapshots, is_prelude_included)
        combined = combined_snapshot_rgbd(self.snapshots, is_prelude_included)
        return torch.from_numpy(rgbd_to_video_array(original)), torch.from_numpy(rgbd_to_video_array(combined))


def run_scenarios_with_snapshots(
    godot_env: AvalonEnv, scenarios: List[Scenario], scenario_path: Path, snapshot_count: int = 10
) -> List[ScenarioObservations]:
    snapshot_actions = cast(List[ScenarioActions], snapshot_count * [SAVE_SNAPSHOT_MESSAGE])
    # TODO hopefully we can get the same video checksums with snapshots as without.
    # could even copy debug file?
    behaviors_with_snapshots = [
        copy_with_evenly_interspersed_snapshots(scene, snapshot_actions) for scene in scenarios
    ]

    return [scenario.run(godot_env, scenario_path) for scenario in behaviors_with_snapshots]


def combined_snapshot_videos(snapshots: List[SnapshotObservations]) -> torch.Tensor:
    combined_observations = flatten([snapshot.action_observations for snapshot in snapshots])
    return torch.from_numpy(observation_video_array(combined_observations))


def compare_all_collections_with_originals(
    collections: List[SnapshotCollection], is_prelude_included: bool = False
) -> Iterable[List[torch.Tensor]]:
    for c in collections:
        original, snapped = c.get_comparable_videos(is_prelude_included)
        diff = original - snapped
        yield [original, snapped, diff]


def summarize_scenario_snapshots(
    snapshots: List[SnapshotObservations], is_testing: bool = False
) -> Dict[str, ManifestValue]:
    combined_rgbd = combined_snapshot_rgbd(snapshots)
    snapshots[0].context

    patterns_with_unreliable_ordering = [b"[ext_resource", b"ExtResource"]

    summary: Dict[str, ManifestValue] = {
        "combined_videos": np_checksum(combined_rgbd),
        "snapshot_scenes": [
            _dir_checksum(snapshot.dir, "*scn", exclude_lines_matching=patterns_with_unreliable_ordering)
            for snapshot in snapshots
        ],
    }

    if not is_testing:
        summary["_percent_drift"] = percent_difference(original_rgbd(snapshots), combined_rgbd)
        count = len(snapshots)
        frames_per = len(snapshots[0].action_observations)
        summary["_snapshot_details"] = f"{count} snapshots {frames_per} frames each"

    return summary


def original_rgbd(snapshots: List[SnapshotObservations], is_prelude_included: bool = False) -> np.ndarray:
    original = rgbd_observations(snapshots[0].original.observations)

    if is_prelude_included:
        return original

    # TODO not sure why this is off by one
    first_snapshot_frame = snapshots[0].context[1] - 1
    return original[first_snapshot_frame:, :, :, :]


def compare_two_collection_rgbd(
    a: SnapshotCollection, b: SnapshotCollection
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    is_prelude_included = False
    a_rgbd = combined_snapshot_rgbd(a.snapshots, is_prelude_included)
    b_rgbd = combined_snapshot_rgbd(b.snapshots, is_prelude_included)
    return a_rgbd, b_rgbd, a_rgbd - b_rgbd


def compare_two_collections(
    a: SnapshotCollection, b: SnapshotCollection
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def as_vid(x: NDArray) -> torch.Tensor:
        return torch.from_numpy(rgbd_to_video_array(x))

    a_rgbd, b_rgbd, diff = compare_two_collection_rgbd(a, b)
    return (as_vid(a_rgbd), as_vid(b_rgbd), as_vid(diff))


def combined_snapshot_rgbd(snapshots: List[SnapshotObservations], is_prelude_included: bool = False) -> np.ndarray:
    combined_snapshot_rgbd = rgbd_observations(flatten([snapshot.action_observations for snapshot in snapshots]))

    if not is_prelude_included:
        return combined_snapshot_rgbd

    # TODO not sure why this is off by one
    first_snapshot_frame = snapshots[0].context[1] - 1

    # a little janky but quick way, to grab same-sized memory without needing original
    unsnapshotted = combined_snapshot_rgbd[:first_snapshot_frame, :, :, :].copy()
    unsnapshotted[:, :, :, :] = 0

    combined_snapshot_rgbd = np.concatenate([unsnapshotted, combined_snapshot_rgbd])
    return combined_snapshot_rgbd


def percent_difference(original_rgbd: np.ndarray, combined_snapshot_rgbd: np.ndarray) -> float:
    differing = np.count_nonzero(original_rgbd - combined_snapshot_rgbd)
    return float(differing) / original_rgbd.size


def verify_snapshot_observations(snapshot: SnapshotObservations):
    assert np.equal(
        rgbd_observations(snapshot.expected_observations), rgbd_observations(snapshot.observations)
    ), f"{snapshot.original.scenario.name} snapshot {snapshot.index} at frame {snapshot.start_frame} failed to produce equal observations"


def observe_snapshots(original: ScenarioObservations, get_env: Callable[[], AvalonEnv]) -> SnapshotCollection:
    snapshot_observations = []
    actions = [a for a in original.scenario.actions if not isinstance(a, int)]
    for current_index, snapshot_context in enumerate(original.snapshots):
        is_last_snapshot = current_index + 1 == len(original.snapshots)
        next_snapshot_frame = None if is_last_snapshot else original.snapshots[current_index + 1][1]
        snapshot_frame = snapshot_context[1]
        chunk_actions = actions[snapshot_frame:next_snapshot_frame]
        chunk_observed = observe_snapshot_chunk(get_env(), snapshot_context, chunk_actions)
        snapshot_observations.append(
            SnapshotObservations(
                original,
                snapshot_context,
                index=current_index,
                end_frame=next_snapshot_frame,
                observations=chunk_observed,
            )
        )
    return SnapshotCollection(snapshot_observations)


def observe_snapshot_chunk(
    env: AvalonEnv, snapshot: SnapshotContext, actions: Sequence[Union[DebugCameraAction, VRActionType]]
) -> List[AvalonObservationType]:
    snapshot_path, _frame, camera_action = snapshot
    obs, _ = env.load_snapshot(snapshot_path)
    observations = [obs]
    if camera_action:
        camera_action = attr.evolve(camera_action, is_frame_advanced=False)
        _discarded = env.debug_act(camera_action)
    for action in actions:
        if isinstance(action, DebugCameraAction):
            obs = env.debug_act(action)
            observations.append(obs)
        elif action in [SAVE_SNAPSHOT_MESSAGE, LOAD_SNAPSHOT_MESSAGE]:
            raise ValueError(f"shouldn't be passing save/load snapshot messages when observing a snapshot chunk")
        else:
            obs, _ = env.act(action)
            observations.append(obs)
    return observations


def copy_with_evenly_interspersed_snapshots(scene: Scenario, snapshots: Sequence[ScenarioActions]) -> Scenario:
    actions = [*scene.actions]
    _act_len = len(actions)
    _snap_len = len(snapshots)
    snapshot_chunk_size = math.floor(len(actions) / len(snapshots))
    index = snapshot_chunk_size - 1
    for snapshot in snapshots:
        actions.insert(index, snapshot)
        index += snapshot_chunk_size
    return attr.evolve(scene, actions=actions)


def _dir_checksum(dir_path: Path, glob: str = "*", exclude_lines_matching: List[bytes] = []) -> str:
    hash_md5 = hashlib.md5()
    for path in sorted(dir_path.glob(glob)):
        with open(path, "rb") as f:

            def read() -> Iterable[bytes]:
                line = f.readline()
                while len(line) != 0:
                    is_excluded = any(ex for ex in exclude_lines_matching if ex in line)
                    if not is_excluded:
                        yield line
                    line = f.readline()

            for chunk in iter(read()):
                hash_md5.update(chunk)
    return hash_md5.hexdigest()
