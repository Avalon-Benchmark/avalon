import json
import os
import shutil
from pathlib import Path
from typing import Any
from typing import ClassVar
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union
from typing import cast

import attr
import numpy as np
import torch
from scipy import stats
from scipy.spatial.transform import Rotation

from avalon.common.utils import dir_checksum
from avalon.common.utils import file_checksum
from avalon.datagen.env_helper import DebugLogLine
from avalon.datagen.env_helper import get_debug_json_logs
from avalon.datagen.env_helper import observation_video_tensor
from avalon.datagen.godot_env.actions import DebugCameraAction
from avalon.datagen.godot_env.actions import VRActionType
from avalon.datagen.godot_env.godot_env import GodotEnv
from avalon.datagen.godot_env.action_log import GodotEnvActionLog
from avalon.datagen.godot_env.observations import AvalonObservationType
from avalon.datagen.godot_generated_types import ACTION_MESSAGE
from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.export import get_agent_export_config
from avalon.datagen.world_creation.constants import IDENTITY_BASIS
from avalon.datagen.world_creation.entities.animals import Animal
from avalon.datagen.world_creation.entities.constants import CANONICAL_FOOD_HEIGHT_ON_TREE
from avalon.datagen.world_creation.entities.constants import FOOD_TREE_VISIBLE_HEIGHT
from avalon.datagen.world_creation.entities.food import Food
from avalon.datagen.world_creation.entities.food import FoodTree
from avalon.datagen.world_creation.entities.item import Item
from avalon.datagen.world_creation.entities.utils import normalized
from avalon.datagen.world_creation.tasks.eat import add_food_and_tree
from avalon.datagen.world_creation.types import WorldType
from avalon.datagen.world_creation.worlds.creation import create_world_for_skill_scenario
from avalon.datagen.world_creation.worlds.export import export_world
from avalon.datagen.world_creation.worlds.height_map import Point3DNP
from avalon.datagen.world_creation.worlds.height_map import _LegacyEdgeNoise
from avalon.datagen.world_creation.worlds.obstacles.configure import make_ring
from avalon.datagen.world_creation.worlds.world import World
from avalon.datagen.world_creation.worlds.world_locations import WorldLocations

AvalonEnv = GodotEnv[AvalonObservationType, VRActionType]
ScenarioActions = Union[DebugCameraAction, VRActionType]


def _export_conf(is_human: bool = False) -> ExportConfig:
    config = attr.evolve(
        get_agent_export_config(),
        is_exported_with_absolute_paths=False,
        is_border_calculation_detailed=False,
        is_legacy_biome_based_on_natural_height_set=False,
    )
    if not is_human:
        return config
    return attr.evolve(
        config,
        name="oculus",
        scenery_mode="tree",
        is_meta_data_exported=True,
    )


def _create_simple_world(
    goal_distance: stats.norm,
    rand: np.random.Generator,
    export_config: ExportConfig,
    difficulty: float = 0.0,
    size_in_meters: float = 50.0,
    ideal_shore_dist_options: Tuple[float, ...] = (4.0, 2.0),
    spawn_point: Optional[Tuple[float, float]] = None,
    is_spawn_hidden: bool = False,
    spawn_facing: Optional[Tuple[float, float]] = (0.0, 0.0),
    cliff_height: Optional[float] = None,
    foliage_density_modifier: float = 0.0,
) -> Tuple[World, WorldLocations]:
    visibility_height = FOOD_TREE_VISIBLE_HEIGHT
    food_height = CANONICAL_FOOD_HEIGHT_ON_TREE
    diversity = 0
    world, locations = create_world_for_skill_scenario(
        rand,
        diversity,
        food_height,
        goal_distance,
        export_config,
        min_size_in_meters=size_in_meters - 1,
        max_size_in_meters=size_in_meters,
        ideal_shore_dist_options=ideal_shore_dist_options,
        visibility_height=visibility_height,
        world_type=WorldType.PLATONIC,
        foliage_density_modifier=foliage_density_modifier,
    )
    locations, world = world.begin_height_obstacles(locations)
    goal = locations.goal
    spawn = locations.spawn
    if spawn_point:
        spawn_point = (float(spawn_point[0]), float(spawn_point[1]))
        y = world.get_height_at(spawn_point) + 2.0
        spawn = np.array([spawn_point[0], y, spawn_point[1]])
    if is_spawn_hidden:
        spawn = np.array([spawn[0], -10, spawn[2]])
    if spawn_facing is not None:
        goal = np.array([spawn_facing[0], world.get_height_at(spawn_facing) + 2.0, spawn_facing[1]])
    locations = attr.evolve(locations, spawn=spawn, goal=goal)
    if cliff_height is not None:
        world = _add_cliff(cliff_height, rand, world, locations)
    world = world.add_spawn(rand, difficulty, locations.spawn, locations.goal, is_visibility_required=True)
    return world, locations


def _add_cliff(height: float, rand: np.random.Generator, world: World, locations: WorldLocations) -> World:
    locations = attr.evolve(locations, spawn=np.array([0.0, 0.0, 0.0]))
    ring_config = make_ring(
        rand,
        1,
        world,
        locations,
        constraint=None,
        gap_distance=0.0,
        height=height,
        traversal_width=0.0,
        inner_traversal_length=0.0,
        is_single_obstacle=True,
        inner_solution=None,
        # always centered around the goal to ensure that it is a pit
        probability_of_centering_on_spawn=1.0,
        outer_traversal_length=0.0,
        max_additional_radius_multiple=1.0,
        is_inside_climbable=True,
    )
    return world.add_height_obstacle(rand, ring_config, locations.island)


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class ScenarioObservations:
    scenario: "Scenario"
    observations: List[AvalonObservationType]
    scene_checksum: str
    scene_path: str
    video_checksum: str
    debug_output_checksum: str

    debug_output: List[DebugLogLine]

    _video: Optional[torch.Tensor] = None

    @property
    def _checksum_dict(self) -> Dict[str, str]:
        return {
            "scene": self.scene_checksum,
            "video": self.video_checksum,
            "debug_output": self.debug_output_checksum,
        }

    @property
    def checksum_summary(self) -> Dict[str, Dict[str, str]]:
        return {f"{self.scenario.name}": self._checksum_dict}

    @property
    def video(self) -> torch.Tensor:
        if self._video is None:
            self._video = observation_video_tensor(self.observations)
        return self._video

    def is_in(self, checksums: Dict[str, Dict[str, str]]) -> bool:
        historical_checksums = checksums[self.scenario.name]
        return self._checksum_dict == historical_checksums


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class BehaviorManifest:
    source_path: Path
    snapshot_commit: str
    checksums: Dict[str, Dict[str, str]]

    @classmethod
    def load(cls, manifest_path: Path) -> "BehaviorManifest":
        with open(manifest_path) as manifest_file:
            manifest: Dict[str, Any] = json.load(manifest_file)
            return BehaviorManifest(
                snapshot_commit=manifest["snapshot_commit"],
                checksums=manifest["checksums"],
                source_path=manifest_path,
            )

    def to_dict(self):
        return {"snapshot_commit": self.snapshot_commit, "checksums": self.checksums}


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Scenario:
    PRE_DEBUG_FRAMES: ClassVar = 1

    name: str
    animal_type: Optional[Union[Type[Animal], Animal]]
    actions: Sequence[ScenarioActions]
    animal_position: Point3DNP = np.zeros(3)
    size_in_meters: int = 200
    detection_radius_override: Optional[float] = None
    spawn_point: Optional[Tuple[float, float]] = None
    is_animal_facing_player: Optional[bool] = None
    is_spawn_hidden: bool = False
    spawn_facing: Optional[Tuple[float, float]] = None
    is_tree_test: bool = False
    cliff_height: Optional[float] = None
    foliage_density_modifier: float = 0.0
    items: Sequence[Union[Item, Tuple[FoodTree, Food]]] = []

    @classmethod
    def look_at(cls, name: str, distance: float):
        node_id = f"{name}__1" if "__" not in name else name
        return DebugCameraAction.isometric(node_id, distance=distance)

    @classmethod
    def inactive(cls, animal_type: Type[Animal], frames: int) -> "Scenario":
        animal_name = animal_type.__name__.lower()
        null_actions = (frames - 1) * [get_vr_action()]
        actions = [cls.look_at(animal_name, 6.0), *null_actions]
        return Scenario(
            f"{animal_name}_inactive",
            animal_type,
            actions,
            detection_radius_override=0.0,
            is_spawn_hidden=True,
        )

    @classmethod
    def pursuit(
        cls,
        kind: str,
        animal_type: Type[Animal],
        player_speed: float,
        spawn_distance: float = 10.0,
        run_frames: int = 200,
        wait_frames: int = 0,
        stop_frames: int = 50,
        camera_distance: float = 8.0,
        is_facing_player: Optional[bool] = False,
        is_zig_zagging: bool = True,
        return_frame: Optional[int] = None,
        return_multiplier: float = 1,
        is_player_facing_away: bool = False,
        final_actions: List[ScenarioActions] = [],
    ) -> "Scenario":
        animal_name = animal_type.__name__.lower()

        if is_player_facing_away:
            player_speed = -player_speed

        wait = wait_frames * [get_vr_action()]
        actions = [cls.look_at(animal_name, camera_distance), *wait]
        zig_zag_frames = int(run_frames / 4.0)
        zig_zag_steps = int(run_frames / zig_zag_frames)
        zig = get_vr_action(head_z=player_speed)
        zag = get_vr_action(head_x=player_speed / 2, head_z=player_speed / 2)
        for zig_zag in range(zig_zag_steps):
            action = zig
            if is_zig_zagging and zig_zag % 2 == 0:
                action = zag
            if return_frame is not None and return_frame < len(actions):
                action = get_vr_action(
                    head_x=return_multiplier * -action.head_x,
                    head_z=return_multiplier * -action.head_z,
                )
            run_actions = zig_zag_frames * [action]
            actions.extend(run_actions)
        actions.extend(stop_frames * [get_vr_action()])
        actions.extend(final_actions)

        # TODO verify expected hit_points_lost_from_enemies at given intervals

        return Scenario(
            f"{animal_name}_{kind}",
            animal_type,
            actions,
            size_in_meters=500,
            spawn_point=(0, spawn_distance),
            is_animal_facing_player=is_facing_player,
            spawn_facing=(0.0, 2.0 * spawn_distance) if is_player_facing_away else (0.0, 0.0),
        )

    @property
    def key(self) -> str:
        return f"{self.name}__0__0_0"

    def export(self, output_path: Path, is_for_recording: bool = False) -> Path:
        rand = np.random.default_rng(0)
        world, locations = _create_simple_world(
            stats.norm(5.0, 1.0),
            rand=rand,
            export_config=_export_conf(is_for_recording),
            size_in_meters=self.size_in_meters,
            spawn_point=self.spawn_point,
            is_spawn_hidden=self.is_spawn_hidden,
            spawn_facing=self.spawn_facing,
            cliff_height=self.cliff_height,
            foliage_density_modifier=self.foliage_density_modifier,
        )
        rotation = IDENTITY_BASIS
        if self.is_animal_facing_player is not None:
            rotation = _facing_2d(self.animal_position, locations.spawn)
        if self.animal_type is not None:
            animal = (
                attr.evolve(self.animal_type, rotation=rotation)
                if isinstance(self.animal_type, Animal)
                else self.animal_type(
                    rotation=rotation,
                    position=self.animal_position,
                    detection_radius_override=self.detection_radius_override,
                )
            )
            world = world.add_item(animal, reset_height_offset=animal.get_offset())
        for item in self.items:
            if isinstance(item, Item):
                offset = item.position[1]
                if offset == 0 and hasattr(item, "get_offset"):
                    offset = item.get_offset()  # type: ignore[attr-defined]
                world = world.add_item(item, reset_height_offset=offset)
            else:
                tree, food = item
                food_height = tree.get_food_height(food)
                x, _, z = food.position
                food_location = np.array([x, world.get_height_at((x, z)) + food_height, z])
                food = attr.evolve(food, position=food_location)
                world = add_food_and_tree(food, locations.spawn, tree, world)
        world, locations = world.end_height_obstacles(locations=locations, is_accessible_from_water=False)
        full_path = output_path / self.key
        _LegacyEdgeNoise.IS_ENABLED = True
        export_world(full_path, rand, world)
        _LegacyEdgeNoise.IS_ENABLED = False

        return full_path

    def observe(self, env: AvalonEnv, output_path: Path) -> List[AvalonObservationType]:
        world_file = output_path / self.key / "main.tscn"
        # TODO figure out how to guarantee complete order invariance for levels run in the same GodotEnv
        # I was trying to ensure invarience in rng, regardless of observation order,
        # but mouse_flee and deer_flee would be off when flee was not the first executed test in the notebook.
        if not env.is_reset_called_already:
            env.reset_nicely_with_specific_world(episode_seed=0, world_path=str(world_file))
        env.reset_nicely_with_specific_world(episode_seed=0, world_path=str(world_file))
        observations = []
        for action in self.actions:
            if isinstance(action, DebugCameraAction):
                obs = env.debug_act(action)
            else:
                obs, _ = env.act(action)
            observations.append(obs)
        return observations[self.PRE_DEBUG_FRAMES :]

    def run(self, env: AvalonEnv, output_path: Path) -> ScenarioObservations:
        try:
            scene_path = self.export(output_path)
        except Exception as e:
            raise ValueError(f"Failed to export scenario {self.name}: {e.args}") from e
        observations = self.observe(env, output_path)

        folder = os.path.join(env.config.dir_root, self.name)
        os.makedirs(folder, exist_ok=True)

        debug_output = get_debug_json_logs(env)
        debug_json_path = os.path.join(folder, "debug.json")
        shutil.move(os.path.join(env.config.dir_root, "000000", "debug.json"), debug_json_path)

        video_npy_path = os.path.join(folder, f"{self.name}_raw_rgbd.npy")
        np.save(video_npy_path, np.stack([o.rgbd for o in observations]))

        obs = ScenarioObservations(
            self,
            observations,
            scene_checksum=dir_checksum(scene_path),
            video_checksum=file_checksum(video_npy_path),
            debug_output_checksum=file_checksum(debug_json_path),
            debug_output=debug_output,
            scene_path=str(scene_path),
        )
        return obs


def get_vr_action(
    head_x: float = 0.0, head_z: float = 0.0, head_yaw: float = 0.0, head_pitch: float = 0.0
) -> VRActionType:
    return VRActionType(
        head_x=head_x,
        head_y=0.0,
        head_z=head_z,
        head_pitch=head_pitch,
        head_yaw=head_yaw,
        head_roll=0.0,
        left_hand_x=0.0,
        left_hand_y=0.0,
        left_hand_z=0.0,
        left_hand_pitch=0.0,
        left_hand_yaw=0.0,
        left_hand_roll=0.0,
        is_left_hand_grasping=0.0,
        right_hand_x=0.0,
        right_hand_y=0.0,
        right_hand_z=0.0,
        right_hand_pitch=0.0,
        right_hand_yaw=0.0,
        right_hand_roll=0.0,
        is_right_hand_grasping=0.0,
        is_jumping=0.0,
    )


def _facing_2d(from_point: Point3DNP, to_point: Point3DNP) -> np.ndarray:
    direction = to_point - from_point
    direction[1] = 0
    direction = -normalized(direction)
    yaw = np.arctan2(direction[0], direction[2])
    return cast(np.ndarray, Rotation.from_euler("y", yaw).as_matrix().flatten())


def read_human_recorded_actions(scenario_name: str) -> Iterable[VRActionType]:
    action_log_path = Path(__file__).parent / f"data/recorded_actions/{scenario_name}.out"
    if not action_log_path.exists():
        error_message = f"{scenario_name} has no recorded actions"
        assert (
            not "PYTEST_CURRENT_TEST" in os.environ
        ), f"cannot run unfinished human-recorded regression test: {error_message }"
        print(f"{error_message}, please record with record_scenario.py")
        return []
    for message in GodotEnvActionLog.parse_message_log(action_log_path.as_posix(), VRActionType):
        if message[0] == ACTION_MESSAGE:
            yield message[1]
        else:
            print(f"{scenario_name} non-action recorded: {message}")
