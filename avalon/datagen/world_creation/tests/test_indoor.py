import filecmp
from pathlib import Path

import attr
import networkx as nx
import numpy as np
import pytest
from deepdiff import DeepDiff

from avalon.common.utils import only
from avalon.contrib.testing_utils import create_temp_file_path
from avalon.contrib.testing_utils import integration_test
from avalon.contrib.testing_utils import slow_integration_test
from avalon.contrib.testing_utils import temp_file_path_
from avalon.contrib.testing_utils import temp_path_
from avalon.contrib.testing_utils import use
from avalon.datagen.errors import ImpossibleWorldError
from avalon.datagen.world_creation.geometry import BuildingTile
from avalon.datagen.world_creation.indoor.building import Building
from avalon.datagen.world_creation.indoor.building import BuildingNavGraph
from avalon.datagen.world_creation.indoor.components import StoryNavGraph
from avalon.datagen.world_creation.indoor.constants import Azimuth
from avalon.datagen.world_creation.indoor.constants import TileIdentity
from avalon.datagen.world_creation.indoor.task_generator import rebuild_with_aligned_entrance
from avalon.datagen.world_creation.tests.fixtures import EMPTY_FILE_CHECKSUM
from avalon.datagen.world_creation.tests.fixtures import ChecksumManifest
from avalon.datagen.world_creation.tests.fixtures import building_
from avalon.datagen.world_creation.tests.fixtures import building_catalog_id_
from avalon.datagen.world_creation.tests.fixtures import buildings_manifest_
from avalon.datagen.world_creation.tests.fixtures import incompatible_building_catalog_id_
from avalon.datagen.world_creation.tests.fixtures import indoor_world_catalog_id_
from avalon.datagen.world_creation.tests.fixtures import indoor_worlds_manifest_
from avalon.datagen.world_creation.tests.fixtures import seed_
from avalon.datagen.world_creation.tests.generate_references import get_file_checksum
from avalon.datagen.world_creation.tests.generate_references import get_path_checksum
from avalon.datagen.world_creation.tests.helpers import add_entrance_if_has_none
from avalon.datagen.world_creation.tests.helpers import compare_files
from avalon.datagen.world_creation.tests.helpers import create_indoor_world
from avalon.datagen.world_creation.tests.helpers import export_building
from avalon.datagen.world_creation.tests.helpers import get_reference_data_path
from avalon.datagen.world_creation.tests.params import BUILDING_CATALOG
from avalon.datagen.world_creation.tests.params import INDOOR_WORLD_CATALOG
from avalon.datagen.world_creation.tests.params import VALID_BUILDING_CATALOG
from avalon.datagen.world_creation.tests.params import IndoorWorldParams
from avalon.datagen.world_creation.worlds.world import build_building


@slow_integration_test
@use(seed_, building_catalog_id_)
def test_building_is_reproducible(seed: int, building_catalog_id: str):
    building_id = 0
    building_name = "building"
    building_config = VALID_BUILDING_CATALOG[building_catalog_id]

    with create_temp_file_path() as export_path_a, create_temp_file_path() as export_path_b:
        rand = np.random.default_rng(seed)
        building = build_building(building_config, building_id, rand)
        export_building(building, building_name, export_path_a)

        rand = np.random.default_rng(seed)
        building = build_building(building_config, building_id, rand)
        export_building(building, building_name, export_path_b)

        assert filecmp.cmp(export_path_a, export_path_b)


@slow_integration_test
@use(temp_file_path_, seed_, building_catalog_id_, buildings_manifest_)
def test_building_matches_reference(
    temp_file_path: Path, seed: int, building_catalog_id: str, buildings_manifest: ChecksumManifest
):
    building_id = 0
    building_config = BUILDING_CATALOG[building_catalog_id]

    generated_building_path = temp_file_path
    rand = np.random.default_rng(seed)
    building_name = f"{building_catalog_id}__{seed}"
    reference_building_checksum = buildings_manifest["checksums"][building_name]
    try:
        building = build_building(building_config, building_id, rand)
        export_building(building, building_name, generated_building_path)
    except Exception as e:
        # Some building configs generate errors that we're currently not fixing for reproducibility reasons -
        # for these, our reference files are empty.
        if reference_building_checksum != EMPTY_FILE_CHECKSUM:
            # However, we don't want to swallow errors that the tests don't anticipate
            raise e
        Path(generated_building_path).write_text("")
    if get_file_checksum(generated_building_path) != reference_building_checksum:
        reference_buildings_path = get_reference_data_path("buildings", buildings_manifest["snapshot_commit"])
        reference_file_path = reference_buildings_path / f"{building_name}.tscn"
        compare_files(generated_building_path, reference_file_path)


@integration_test
@use(seed_, incompatible_building_catalog_id_)
def test_incompatible_building_raises(seed: int, incompatible_building_catalog_id: str):
    building_id = 0
    incompatible_building_config = BUILDING_CATALOG[incompatible_building_catalog_id]
    rand = np.random.default_rng(seed)
    with pytest.raises(ImpossibleWorldError):
        build_building(incompatible_building_config, building_id, rand)


@slow_integration_test
@use(temp_path_, indoor_world_catalog_id_, indoor_worlds_manifest_)
def test_indoor_world_matches_reference(
    temp_path: Path, indoor_world_catalog_id: str, indoor_worlds_manifest: ChecksumManifest
):
    indoor_world_params: IndoorWorldParams = INDOOR_WORLD_CATALOG[indoor_world_catalog_id]
    create_indoor_world(*indoor_world_params, export_path=temp_path)
    generated_world_checksum = get_path_checksum(temp_path)
    reference_world_checksum = indoor_worlds_manifest["checksums"][indoor_world_catalog_id]
    if generated_world_checksum != reference_world_checksum:
        reference_world_path = get_reference_data_path("indoor_worlds", indoor_worlds_manifest["snapshot_commit"])
        for generated_file_path in temp_path.iterdir():
            reference_file_path = (
                reference_world_path / indoor_world_catalog_id / generated_file_path.relative_to(temp_path)
            )
            compare_files(generated_file_path, reference_file_path)
    else:
        assert generated_world_checksum == reference_world_checksum, "Generated world checksum does not match manifest"


@use(building_)
def test_all_rooms_accessible(building: Building):
    # 1. Check that rooms are connected via hallways and that stories are connected by links
    building_graph = BuildingNavGraph(building)
    room_clusters = nx.connected_components(building_graph)
    ground_story = building.stories[0]
    first_story_nodes = {building_graph.get_room_node(ground_story, room) for room in ground_story.rooms}
    ground_story_clusters = [c for c in room_clusters if c.intersection(first_story_nodes)]
    primary_cluster = sorted(ground_story_clusters, key=lambda c: len(c))[-1]

    for story in building.stories:
        for room in story.rooms:
            room_node = building_graph.get_room_node(story, room)
            try:
                assert room_node in primary_cluster
            except AssertionError as e:
                is_decorative_story = story.num > 0 and len(story.story_links) == 0
                if not is_decorative_story:
                    raise e

        # 2. Check that each story is traversable on a tile-by-tile level (no blocked hallways/entrances/links)
        story_graph = StoryNavGraph(story)
        tile_clusters = list(nx.connected_components(story_graph))
        assert len(tile_clusters) == 1


@use(seed_, building_)
def test_entrance_built(seed: int, building: Building):
    add_entrance_if_has_none(building, np.random.default_rng(seed))
    for story in building.stories:
        tiles = building.generate_tiles(story.num)
        for entrance in story.entrances:
            points_in_outline = entrance.get_points_in_story_outline(story)
            for point in points_in_outline:
                assert TileIdentity(tiles[point[0], point[1]]) == TileIdentity.HALLWAY


@use(seed_, building_)
def test_entrance_unobstructed(seed: int, building: Building):
    add_entrance_if_has_none(building, np.random.default_rng(seed))
    for story in building.stories:
        tiles = building.generate_tiles(story.num)
        for entrance in story.entrances:
            connected_room, landing_position_in_room_space = entrance.get_connected_room_and_landing_position(story)
            landing_position = BuildingTile(
                landing_position_in_room_space.x + connected_room.position.x,
                landing_position_in_room_space.z + connected_room.position.z,
            )
            assert TileIdentity(tiles[landing_position.z, landing_position.x]) == TileIdentity.ROOM


@use(seed_, building_)
def test_rebuild_with_aligned_entrance(seed: int, building: Building):
    rand = np.random.default_rng(seed)
    aligned_building = rebuild_with_aligned_entrance(building, rand, 0)
    entrance = only(aligned_building.stories[0].entrances)
    assert entrance.azimuth == Azimuth.EAST


def assert_buildings_equal(building_a: Building, building_b: Building) -> None:
    dict_a = attr.asdict(building_a)
    dict_b = attr.asdict(building_b)
    assert DeepDiff(dict_a, dict_b, math_epsilon=1e-15) == {}


@use(building_)
def test_rebuild_rotated(building: Building):
    # Back and forth
    for rotation in {0, 90, -90, 180}:
        assert_buildings_equal(building, building.rebuild_rotated(rotation).rebuild_rotated(-rotation))

    # Full circle
    assert_buildings_equal(building, building.rebuild_rotated(180).rebuild_rotated(180))
    for rotation in {90, -90}:
        rotated_building = building
        for i in range(4):
            rotated_building = rotated_building.rebuild_rotated(rotation)
        assert_buildings_equal(building, rotated_building)
