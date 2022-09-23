import json
from pathlib import Path
from typing import Dict
from typing import Optional
from typing import TypedDict
from typing import cast

import numpy as np
from loguru import logger

from avalon.contrib.testing_utils import RequestFixture
from avalon.contrib.testing_utils import fixture
from avalon.contrib.testing_utils import use
from avalon.datagen.world_creation.indoor.building import Building
from avalon.datagen.world_creation.tests.params import CANONICAL_SEED
from avalon.datagen.world_creation.tests.params import INDOOR_WORLD_CATALOG
from avalon.datagen.world_creation.tests.params import INVALID_BUILDING_CATALOG
from avalon.datagen.world_creation.tests.params import OUTDOOR_WORLD_CATALOG
from avalon.datagen.world_creation.tests.params import VALID_BUILDING_CATALOG
from avalon.datagen.world_creation.worlds.world import build_building

EMPTY_FILE_CHECKSUM = "d41d8cd98f00b204e9800998ecf8427e"


@fixture
def seed_() -> int:
    return CANONICAL_SEED


@fixture(params=VALID_BUILDING_CATALOG.keys())
def building_catalog_id_(request: RequestFixture) -> str:
    return cast(str, request.param)


@fixture(params=VALID_BUILDING_CATALOG.items())
@use(seed_)
def building_(request: RequestFixture, seed: int) -> Building:
    building_catalog_id, building_config = request.param
    logger.info(f"Generating building {building_catalog_id}")
    rand = np.random.default_rng(seed)
    return build_building(building_config, building_catalog_id, rand)


@fixture(params=INVALID_BUILDING_CATALOG.keys())
def incompatible_building_catalog_id_(request: RequestFixture) -> str:
    return cast(str, request.param)


@fixture(params=INDOOR_WORLD_CATALOG.keys())
def indoor_world_catalog_id_(request: RequestFixture) -> str:
    return cast(str, request.param)


class ChecksumManifest(TypedDict):
    snapshot_commit: str
    checksums: Dict[str, str]


def get_current_reference_manifest(reference_name: str, data_path: Optional[Path] = None) -> ChecksumManifest:
    if data_path is None:
        data_path = Path(__file__).parent / "data"
    return cast(ChecksumManifest, json.load(open(data_path / f"{reference_name}_manifest.json", "r")))


@fixture(scope="session")
def buildings_manifest_() -> ChecksumManifest:
    return get_current_reference_manifest("buildings")


@fixture(scope="session")
def indoor_worlds_manifest_() -> ChecksumManifest:
    return get_current_reference_manifest("indoor_worlds")


@fixture(params=OUTDOOR_WORLD_CATALOG.keys())
def outdoor_world_catalog_id_(request: RequestFixture) -> str:
    return cast(str, request.param)


@fixture(scope="session")
def outdoor_worlds_manifest_() -> ChecksumManifest:
    return get_current_reference_manifest("outdoor_worlds")
