# run with `pytest  --noconftest -k test_server_state`
import gzip
import json
import random
import uuid
from typing import List
from unittest.mock import patch

import pytest

from avalon.contrib.utils import temp_dir

# note: using relative imports because this server runs independent from the rest of standalone/avalon
from .server_state import APK_VERSIONS_PATH
from .server_state import IGNORED_MARKER
from .server_state import METADATA_FILE
from .server_state import RESET_MARKER
from .server_state import RunID
from .server_state import ServerState
from .server_state import UserID
from .server_state import WorldID
from .server_state import get_current_server_state

TEST_ROOT_PATH = "/tmp/avalon_server_test"
TEST_USERS = [f"user_{i}" for i in range(5)]
TEST_FILENAME = "test.out"

USER_ID = "user_1"
WORLD_ID = "world_1"
IGNORED_WORLD = "world_2"
NEXT_WORLD = "world_3"
PRACTICE_WORLD = "practice_world_4"

RUN_ID = str(uuid.uuid4())
NEXT_RUN_ID = str(uuid.uuid4())

TEST_WORLDS = [WORLD_ID, IGNORED_WORLD, NEXT_WORLD, PRACTICE_WORLD]

API_VERSION = "test_api_version"
APK_VERSION = "test_apk_version"
INVALID_APK_VERSION = "invalid_apk_version"


@pytest.fixture()
def server_state():
    with temp_dir(TEST_ROOT_PATH) as root_path:
        for world_id in TEST_WORLDS:
            path = root_path / world_id
            if not path.exists():
                path.mkdir()

        server_state = get_current_server_state(root_path, API_VERSION)
        yield server_state


@pytest.fixture()
def run_id():
    return RUN_ID


@pytest.fixture()
def next_run_id():
    return NEXT_RUN_ID


@pytest.fixture()
def world_id():
    return WORLD_ID


@pytest.fixture()
def ignored_world_id():
    return IGNORED_WORLD


@pytest.fixture()
def next_world_id():
    return NEXT_WORLD


@pytest.fixture()
def practice_world_id():
    return PRACTICE_WORLD


@pytest.fixture()
def user_id():
    return USER_ID


@pytest.fixture()
def users():
    return TEST_USERS


@pytest.fixture()
def filename():
    return TEST_FILENAME


@pytest.fixture()
def apk_version():
    return APK_VERSION


@pytest.fixture()
def invalid_apk_version():
    return INVALID_APK_VERSION


def _fill_server_state(
    server_state: ServerState,
    apk_version: str,
    world_id: WorldID,
    ignored_world_id: WorldID,
    users: List[UserID],
    run_id: RunID,
    is_success: bool,
):
    for user_id in users:
        server_state.record_user_start_on_world(world_id=world_id, user_id=user_id, run_id=run_id)

        # mock metadata json file
        user_path = server_state.get_user_path_from_user_id_for_world(
            world_id=world_id, user_id=user_id, run_id=run_id
        )
        parent_path = user_path / apk_version
        if not parent_path.exists():
            parent_path.mkdir(parents=True)
        metadata_path = parent_path / METADATA_FILE
        with gzip.open(metadata_path, "wb") as f:
            f.write(json.dumps({"is_success": is_success}).encode("utf-8"))

    server_state.ignore_world(world_id=ignored_world_id)

    server_state.add_apk_version(apk_version=apk_version)

    return server_state


@pytest.fixture()
def filled_server_state(
    server_state: ServerState,
    apk_version: str,
    world_id: WorldID,
    ignored_world_id: WorldID,
    users: List[UserID],
    run_id: RunID,
):
    yield _fill_server_state(
        server_state,
        apk_version,
        world_id,
        ignored_world_id,
        users,
        run_id,
        is_success=True,
    )


@pytest.fixture()
def filled_server_state_with_no_success(
    server_state: ServerState,
    apk_version: str,
    world_id: WorldID,
    ignored_world_id: WorldID,
    users: List[UserID],
    run_id: RunID,
):
    yield _fill_server_state(
        server_state,
        apk_version,
        world_id,
        ignored_world_id,
        users,
        run_id,
        is_success=False,
    )


def test_valid_apk_version(server_state: ServerState, apk_version: str) -> None:
    assert not server_state.valid_apk_versions
    server_state.add_apk_version(apk_version=apk_version)
    assert server_state.is_apk_version_valid(apk_version=apk_version)
    assert (server_state.root_path / APK_VERSIONS_PATH / f"{apk_version}.marker").exists()


def test_verify_apk_version(filled_server_state: ServerState, apk_version: str, invalid_apk_version: str) -> None:
    assert filled_server_state.is_apk_version_valid(apk_version=apk_version)
    assert not filled_server_state.is_apk_version_valid(apk_version=invalid_apk_version)


def test_world_blacklist(server_state: ServerState, ignored_world_id: WorldID, run_id: RunID) -> None:
    server_state.ignore_world(world_id=ignored_world_id)
    assert ignored_world_id not in server_state.get_available_worlds()
    assert ignored_world_id in server_state.ignored_world_ids

    world_path = server_state.get_world_path_from_world_id(world_id=ignored_world_id)
    ignored_path = world_path / IGNORED_MARKER
    assert ignored_path.exists()


def test_user_start_world(server_state: ServerState, world_id: WorldID, user_id: UserID, run_id: RunID) -> None:
    server_state.record_user_start_on_world(world_id=world_id, user_id=user_id, run_id=run_id)
    assert user_id in server_state.user_ids_by_world_id[world_id]
    assert server_state.did_user_start_world(world_id=world_id, user_id=user_id, run_id=run_id)
    user_path = server_state.get_user_path_from_user_id_for_world(world_id=world_id, user_id=user_id, run_id=run_id)
    assert user_path.exists()


def test_user_reset_world(server_state: ServerState, world_id: WorldID, user_id: UserID, run_id: RunID) -> None:
    server_state.record_user_start_on_world(world_id=world_id, user_id=user_id, run_id=run_id)
    server_state.record_user_reset_on_world(world_id=world_id, user_id=user_id, run_id=run_id)
    assert user_id in server_state.resets_by_world_id[world_id]
    assert server_state.did_user_reset_world(world_id=world_id, user_id=user_id, run_id=run_id)

    user_path = server_state.get_user_path_from_user_id_for_world(world_id=world_id, user_id=user_id, run_id=run_id)
    reset_path = user_path / RESET_MARKER
    assert reset_path.exists()


def test_user_save_data(
    server_state: ServerState, apk_version: str, world_id: WorldID, user_id: UserID, run_id: RunID, filename: str
) -> None:
    server_state.record_user_start_on_world(world_id=world_id, user_id=user_id, run_id=run_id)
    server_state.save_user_data_for_world(
        data=bytes(), filename=filename, apk_version=apk_version, world_id=world_id, user_id=user_id, run_id=run_id
    )

    assert server_state.did_user_finish_world(world_id=world_id, user_id=user_id, run_id=run_id)

    user_path = server_state.get_user_path_from_user_id_for_world(world_id=world_id, user_id=user_id, run_id=run_id)
    file_path = user_path / apk_version / f"{filename}.gz"
    assert file_path.exists()


def test_available_worlds(
    filled_server_state: ServerState, world_id: WorldID, ignored_world_id: WorldID, users: List[UserID]
) -> None:
    num_valid_plays = filled_server_state.get_num_valid_plays_for_world(world_id=world_id)
    assert num_valid_plays == len(users)
    assert world_id not in filled_server_state.get_available_worlds()
    assert ignored_world_id not in filled_server_state.get_available_worlds()


def test_available_worlds_with_no_success(
    filled_server_state_with_no_success: ServerState, world_id: WorldID, ignored_world_id: WorldID, users: List[UserID]
) -> None:
    num_valid_plays = filled_server_state_with_no_success.get_num_valid_plays_for_world(world_id=world_id)
    assert num_valid_plays == len(users)
    assert world_id in filled_server_state_with_no_success.get_available_worlds()
    assert ignored_world_id not in filled_server_state_with_no_success.get_available_worlds()


def test_practice_worlds(
    server_state: ServerState, practice_world_id: WorldID, user_id: UserID, run_id: RunID, next_run_id: RunID
) -> None:
    server_state.record_user_start_on_world(world_id=practice_world_id, user_id=user_id, run_id=run_id)
    assert user_id in server_state.user_ids_by_world_id[practice_world_id]
    # this normally fails if we already started on a world
    server_state.record_user_start_on_world(world_id=practice_world_id, user_id=user_id, run_id=next_run_id)
    assert practice_world_id not in server_state.get_available_worlds()


def test_available_worlds_with_reset(
    filled_server_state: ServerState,
    world_id: WorldID,
    ignored_world_id: WorldID,
    users: List[UserID],
    user_id: UserID,
    run_id: RunID,
) -> None:
    filled_server_state.record_user_reset_on_world(world_id=world_id, user_id=user_id, run_id=run_id)
    assert user_id in filled_server_state.resets_by_world_id[world_id]
    num_valid_plays = filled_server_state.get_num_valid_plays_for_world(world_id=world_id)
    assert num_valid_plays == len(users) - 1
    assert world_id in filled_server_state.get_available_worlds()
    assert ignored_world_id not in filled_server_state.get_available_worlds()


def test_available_worlds_for_user(
    filled_server_state: ServerState,
    world_id: WorldID,
    ignored_world_id: WorldID,
    next_world_id: WorldID,
    users: List[UserID],
    user_id: UserID,
    run_id: RunID,
) -> None:
    filled_server_state.record_user_start_on_world(world_id=next_world_id, user_id=user_id, run_id=run_id)
    available_worlds = filled_server_state.get_worlds_available_to_user(user_id=user_id)

    assert world_id not in available_worlds
    assert ignored_world_id not in available_worlds
    assert next_world_id not in available_worlds


def test_world_sample(
    filled_server_state: ServerState,
    world_id: WorldID,
    ignored_world_id: WorldID,
    next_world_id: WorldID,
    user_id: UserID,
) -> None:
    with patch.object(random, "sample", return_value=[next_world_id]) as mock_method:
        actual_next_world_id = filled_server_state.get_world_for_user(user_id=user_id)
    assert actual_next_world_id
    assert actual_next_world_id == next_world_id
    assert actual_next_world_id != world_id
    assert actual_next_world_id != ignored_world_id
