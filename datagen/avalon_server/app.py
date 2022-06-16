#!/usr/bin/env python3
import gzip
import hashlib
import json
import multiprocessing
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set

import attr
import flask
import sentry_sdk
from flask import send_from_directory
from sentry_sdk.integrations.flask import FlaskIntegration

DEFAULT_ROOT_PATH = "/tmp/avalon_server"
WorldID = str
UserID = str
RunID = str

RESET_MARKER = "reset.marker"
IGNORED_MARKER = "ignored.marker"
CRASH_FOLDER = "crash"
MAX_NUM_PLAYS = 5
ACTION_FILE = "actions.out.gz"
OBSERVATION_FILE = "observations.out.gz"
HUMAN_INPUT_FILE = "human_inputs.out.gz"
METADATA_FILE = "metadata.json.gz"
IGNORED_USER_IDS = {"ignored.marker"}
APK_DIR = "/tmp/avalon/apks"
APK_VERSIONS_PATH = "versions/"
IS_SENTRY_ENABLED = False
API_VERSION = os.getenv("API_VERSION")
ROOT_PATH = Path(os.getenv("ROOT_PATH")) / API_VERSION
WORLD_ROOT = ROOT_PATH / "worlds"
WORLD_ROOT.mkdir(mode=0o755, exist_ok=True, parents=True)

if IS_SENTRY_ENABLED:
    sentry_sdk.init(
        dsn="<your sentry dsn goes here>",
        integrations=[FlaskIntegration()],
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        # We recommend adjusting this value in production.
        traces_sample_rate=1.0,
    )


def create_app():
    app = flask.Flask("avalon_server")

    if ROOT_PATH and not Path(ROOT_PATH).exists():
        Path(ROOT_PATH).mkdir(parents=True)
    return app


app = create_app()
lock = multiprocessing.Lock()


class UserDoesNotExistInWorld(Exception):
    pass


class DuplicateUserDataInWorld(Exception):
    pass


class WorldDoesNotExist(Exception):
    pass


def sample_world_id(world_ids: Sequence[WorldID]) -> WorldID:
    return random.sample(world_ids, 1)[0]


def get_world_id_from_world_path(world_path: Path) -> WorldID:
    return str(world_path.parts[-1])


def get_user_id_from_user_path(user_path: Path) -> UserID:
    return str(user_path.parts[-1])


def is_practice_world(world_id: str) -> bool:
    return world_id.startswith("practice")


def is_run_id_crash(run_id: str) -> bool:
    return run_id.startswith(CRASH_FOLDER)


def get_user_path_with_apk_version(user_path: Path, apk_versions: Set[str]) -> Optional[Path]:
    children = []
    crashes = []
    for child_dir in user_path.iterdir():
        if child_dir.is_dir():
            if child_dir.name in apk_versions:
                children.append(child_dir)
            if child_dir.name == CRASH_FOLDER:
                crashes.append(child_dir)
    # user crashed or they're in progress
    if len(crashes) > 0 or len(children) == 0:
        return None
    assert len(children) == 1, f"Found more paths than expected {children}"
    return children[0]


def read_metadata(user_world_path: Path, apk_versions: Set[str]) -> Optional[Dict]:
    user_path_with_apk_version = get_user_path_with_apk_version(user_world_path, apk_versions)
    if not user_path_with_apk_version:
        return None
    metadata_path = user_path_with_apk_version / METADATA_FILE
    if not metadata_path.exists():
        return None
    with gzip.open(metadata_path, "r") as fin:
        json_bytes = fin.read()
    json_str = json_bytes.decode("utf-8")
    return json.loads(json_str)


def read_world_metadata(apk_version: str, world_id: str) -> Optional[Dict]:
    world_metadata_path = WORLD_ROOT / apk_version / (world_id + ".json")
    if world_metadata_path.exists():
        return json.loads(world_metadata_path.read_text())
    else:
        return None


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ServerState:
    root_path: Path
    api_version: str
    valid_apk_versions: Set[str]
    all_world_ids: Set[WorldID]
    user_ids_by_world_id: Dict[WorldID, Set[UserID]]
    resets_by_world_id: Dict[WorldID, Set[UserID]]
    ignored_world_ids: Set[WorldID]

    def add_apk_version(self, apk_version: str):
        apk_version_path = self.root_path / APK_VERSIONS_PATH / f"{apk_version}.marker"
        apk_version_path.touch(exist_ok=True)
        self.valid_apk_versions.add(apk_version)

    def is_apk_version_valid(self, apk_version: str) -> bool:
        return apk_version in self.valid_apk_versions

    def is_user_on_world(
        self,
        world_id: WorldID,
        user_id: UserID,
    ) -> bool:
        if world_id not in self.user_ids_by_world_id:
            raise WorldDoesNotExist(f"Could not find {world_id} in state")
        return user_id in self.user_ids_by_world_id[world_id]

    def get_all_user_ids(self) -> Set[UserID]:
        all_user_ids = set()
        for user_ids in self.user_ids_by_world_id.values():
            for user_id in user_ids:
                if user_id not in IGNORED_USER_IDS:
                    all_user_ids.add(user_id)
        return all_user_ids

    def get_world_ids_by_user_id(self, user_id: UserID, is_including_practice: bool = True) -> Set[WorldID]:
        world_ids = set()
        for world_id, user_ids in self.user_ids_by_world_id.items():
            if not is_including_practice and is_practice_world(world_id=world_id):
                continue
            if user_id in user_ids:
                world_ids.add(world_id)
        return world_ids

    def get_world_ids_by_user_ids(self, is_including_practice: bool = True) -> Dict[UserID, Set[WorldID]]:
        world_ids_by_user_ids = defaultdict(set)
        for world_id, user_ids in self.user_ids_by_world_id.items():
            if not is_including_practice and is_practice_world(world_id=world_id):
                continue
            for user_id in user_ids:
                world_ids_by_user_ids[user_id].add(world_id)
        return world_ids_by_user_ids

    def get_most_recently_visited_world_by_user(self, user_id: UserID) -> Optional[WorldID]:
        world_ids = self.get_world_ids_by_user_id(user_id=user_id)

        min_start_time = 0
        recent_world_id = None

        for world_id in world_ids:
            if is_practice_world(world_id=world_id):
                continue

            user_world_path = self.get_user_path_from_user_id_for_world(world_id=world_id, user_id=user_id)
            metadata = read_metadata(user_world_path, apk_versions=self.valid_apk_versions)

            # if there's no metadata that crash gets logged elsewhere
            if metadata and metadata["start_time"] > min_start_time:
                min_start_time = metadata["start_time"]
                recent_world_id = world_id

        return recent_world_id

    def get_num_crashes_for_world(self, world_id: WorldID) -> int:
        world_path = self.get_world_path_from_world_id(world_id=world_id)
        num_crashes = 0
        for dir in world_path.iterdir():
            if CRASH_FOLDER in dir.name:
                num_crashes += 1
        return num_crashes

    def get_num_valid_plays_for_world(self, world_id: WorldID) -> int:
        num_users_started_world = len(list(self.user_ids_by_world_id[world_id]))
        num_resets_on_world = len(list(self.resets_by_world_id[world_id]))
        num_crashes_on_world = self.get_num_crashes_for_world(world_id)
        num_valid_plays = num_users_started_world - num_resets_on_world - num_crashes_on_world
        return num_valid_plays

    def get_available_worlds(self, is_limiting_plays: bool = True) -> List[WorldID]:
        world_ids = []
        for world_id in self.all_world_ids:
            num_valid_plays = self.get_num_valid_plays_for_world(world_id=world_id)
            if (
                world_id not in self.ignored_world_ids
                and (num_valid_plays < MAX_NUM_PLAYS or not is_limiting_plays)
                and not is_practice_world(world_id=world_id)
            ):
                world_ids.append(world_id)
        return world_ids

    def get_worlds_available_to_user(self, user_id: UserID, is_limiting_plays: bool = True) -> List[WorldID]:
        return [
            world_id
            for world_id in self.get_available_worlds(is_limiting_plays=is_limiting_plays)
            if not self.is_user_on_world(world_id=world_id, user_id=user_id)
            and not is_practice_world(world_id=world_id)
        ]

    def get_world_for_user(self, user_id: UserID) -> Optional[WorldID]:
        potential_world_ids = self.get_worlds_available_to_user(user_id=user_id)
        if len(potential_world_ids) == 0:
            non_play_limited_world_ids = self.get_worlds_available_to_user(user_id=user_id, is_limiting_plays=False)
            return sample_world_id(non_play_limited_world_ids)
        else:
            return sample_world_id(potential_world_ids)

    def record_user_start_on_world(self, world_id: WorldID, user_id: UserID, run_id: RunID):
        user_path = self.get_user_path_from_user_id_for_world(world_id=world_id, user_id=user_id, run_id=run_id)

        if user_path.exists():
            raise DuplicateUserDataInWorld(f"User {user_id} already has data in {world_id}")

        user_path.mkdir(parents=True)

        if world_id not in self.user_ids_by_world_id:
            raise WorldDoesNotExist(f"Could not find {world_id}")

        self.user_ids_by_world_id[world_id].add(user_id)

    def record_user_reset_on_world(self, world_id: WorldID, user_id: UserID, run_id: RunID):
        user_path = self.get_user_path_from_user_id_for_world(world_id=world_id, user_id=user_id, run_id=run_id)

        if not user_path.exists():
            raise UserDoesNotExistInWorld(f"User {user_id} is not on {world_id}")

        reset_path = user_path / RESET_MARKER
        reset_path.touch()

        if world_id not in self.user_ids_by_world_id:
            raise WorldDoesNotExist(f"Could not find {world_id}")

        self.resets_by_world_id[world_id].add(user_id)

    def save_user_data_for_world(
        self,
        data: bytes,
        filename: str,
        apk_version: str,
        world_id: WorldID,
        user_id: UserID,
        run_id: RunID,
    ):
        user_path = self.get_user_path_from_user_id_for_world(world_id=world_id, user_id=user_id, run_id=run_id)

        if is_run_id_crash(run_id=run_id):
            if not user_path.exists():
                user_path.mkdir(parents=True)
        else:
            if not user_path.exists():
                raise UserDoesNotExistInWorld(f"User {user_id} is not on {world_id}")

        if world_id not in self.user_ids_by_world_id:
            raise WorldDoesNotExist(f"Could not find {world_id}")

        user_with_apk_version_path = Path(user_path / apk_version)
        user_with_apk_version_path.mkdir(exist_ok=True)
        file_path = user_with_apk_version_path / f"{filename}.gz"
        file_path.write_bytes(data)

    def ignore_world(self, world_id: WorldID):
        world_path = self.get_world_path_from_world_id(world_id=world_id)
        ignored_marker_path = world_path / IGNORED_MARKER
        if not ignored_marker_path.exists():
            ignored_marker_path.touch()
        self.ignored_world_ids.add(world_id)

    def get_world_path_from_world_id(self, world_id: WorldID) -> Path:
        return self.root_path / world_id

    def get_user_path_from_user_id_for_world(
        self, world_id: WorldID, user_id: UserID, run_id: Optional[RunID] = None
    ) -> Path:
        world_path = self.get_world_path_from_world_id(world_id=world_id)
        if run_id and (is_practice_world(world_id=world_id) or is_run_id_crash(run_id=run_id)):
            return world_path / user_id / run_id
        else:
            return world_path / user_id

    def did_user_start_world(self, world_id: WorldID, user_id: UserID, run_id: RunID) -> bool:
        user_path = self.get_user_path_from_user_id_for_world(world_id=world_id, user_id=user_id, run_id=run_id)
        return user_path.exists()

    def did_user_reset_world(self, world_id: WorldID, user_id: UserID, run_id: RunID) -> bool:
        user_path = self.get_user_path_from_user_id_for_world(world_id=world_id, user_id=user_id, run_id=run_id)
        reset_path = user_path / RESET_MARKER
        return user_path.exists() and reset_path.exists()

    def did_user_finish_world(self, world_id: WorldID, user_id: UserID, run_id: RunID) -> bool:
        user_path = self.get_user_path_from_user_id_for_world(world_id=world_id, user_id=user_id, run_id=run_id)
        return (
            self.did_user_start_world(world_id=world_id, user_id=user_id, run_id=run_id)
            and not self.did_user_reset_world(world_id=world_id, user_id=user_id, run_id=run_id)
            and len(list(user_path.iterdir())) > 0
        )


def get_active_time_played_by_users(server_state: ServerState):
    time_in_hours_by_user_id = defaultdict(float)
    broken_user_paths_and_ids = []
    for user_id, world_ids in server_state.get_world_ids_by_user_ids(is_including_practice=False).items():
        for world_id in world_ids:
            if user_id not in ["ignored.marker"]:
                user_path = server_state.get_user_path_from_user_id_for_world(
                    world_id=world_id, user_id=user_id, run_id=None
                )
                metadata = read_metadata(user_path, server_state.valid_apk_versions)
                if not metadata:
                    broken_user_paths_and_ids.append((user_id, user_path))
                    continue
                time_in_hours_by_user_id[user_id] += (metadata["end_time"] - metadata["start_time"]) / 3600
    return {
        k: v for k, v in sorted(time_in_hours_by_user_id.items(), key=lambda item: -item[1])
    }, broken_user_paths_and_ids


def get_current_server_state(root_path: Path) -> ServerState:
    # print(f"Getting current server state from {root_path}")
    user_ids_by_world_id = {}
    resets_by_world_id = {}
    ignored_world_ids = set()
    valid_apk_versions = set()

    apk_versions_path = root_path / APK_VERSIONS_PATH
    apk_versions_path.mkdir(exist_ok=True)

    for path in apk_versions_path.iterdir():
        valid_apk_versions.add(path.name.replace(".marker", ""))

    for world_path in root_path.iterdir():
        if world_path == apk_versions_path or world_path == WORLD_ROOT:
            continue

        world_id = get_world_id_from_world_path(world_path)

        # some worlds might be super bugged so we'll want to manually black list them
        ignored_marker_path = world_path / IGNORED_MARKER
        if ignored_marker_path.exists():
            ignored_world_ids.add(world_id)

        user_ids_by_world_id[world_id] = set()
        resets_by_world_id[world_id] = set()

        for user_path in world_path.iterdir():
            # all user paths must be directories
            if user_path.name == IGNORED_MARKER or not user_path.is_dir():
                continue
            user_id = get_user_id_from_user_path(user_path)
            if user_id in user_ids_by_world_id[world_id]:
                raise DuplicateUserDataInWorld(f"User {user_id} already has data in {world_id}")
            user_ids_by_world_id[world_id].add(user_id)

            reset_marker_path = user_path / RESET_MARKER
            if reset_marker_path.exists():
                resets_by_world_id[world_id].add(user_id)

    return ServerState(
        root_path=root_path,
        api_version=str(API_VERSION),
        valid_apk_versions=valid_apk_versions,
        all_world_ids=set(user_ids_by_world_id.keys()),
        user_ids_by_world_id=user_ids_by_world_id,
        resets_by_world_id=resets_by_world_id,
        ignored_world_ids=ignored_world_ids,
    )


@app.route("/add_apk_version/<apk_version>/", methods=["GET"])
def add_apk_version(apk_version: str):
    with lock:
        state = get_current_server_state(ROOT_PATH)
        state.add_apk_version(apk_version=apk_version)
    return flask.jsonify(
        {
            "headers": dict(flask.request.headers),
            "apk_versions": list(state.valid_apk_versions),
        }
    )


@app.route("/get_apk_versions/", methods=["GET"])
def get_apk_versions():
    with lock:
        state = get_current_server_state(ROOT_PATH)
    return flask.jsonify(
        {
            "headers": dict(flask.request.headers),
            "apk_versions": list(state.valid_apk_versions),
        }
    )


@app.route("/verify/<apk_version>/", methods=["GET"])
def verify(apk_version: str):
    with lock:
        state = get_current_server_state(ROOT_PATH)
        is_valid = apk_version in state.valid_apk_versions
    return flask.jsonify(
        {
            "headers": dict(flask.request.headers),
            "is_valid": is_valid,
        }
    )


@app.route("/info/", methods=["GET"])
def info():
    return flask.jsonify(
        {
            "headers": dict(flask.request.headers),
            "root_path": str(ROOT_PATH),
        }
    )


@app.route("/make_world/<world_id>/", methods=["GET"])
def make_world(world_id: WorldID):
    with lock:
        world_path = ROOT_PATH / world_id
        is_already_present = world_path.exists()
        if not is_already_present:
            world_path.mkdir()
    return flask.jsonify(
        {
            "headers": dict(flask.request.headers),
            "world_path": str(world_path),
            "is_already_present": int(is_already_present),
        }
    )


@app.route("/get_state/", methods=["GET"])
def get_state():
    with lock:
        state = get_current_server_state(ROOT_PATH)
        serialized_state = attr.asdict(state)
        serialized_state.pop("root_path")
    return flask.jsonify(
        {
            "headers": dict(flask.request.headers),
            "root_path": str(state.root_path),
            "state": serialized_state,
        }
    )


@app.route("/users/", methods=["GET"])
def users():
    with lock:
        state = get_current_server_state(ROOT_PATH)
        user_ids = list(state.get_all_user_ids())
    return flask.jsonify(
        {
            "headers": dict(flask.request.headers),
            "user_ids": user_ids,
        }
    )


@app.route("/time_played_by_user/", methods=["GET"])
def time_played_by_user():
    with lock:
        state = get_current_server_state(ROOT_PATH)
        time_played_by_user, _ = get_active_time_played_by_users(server_state=state)
    return flask.jsonify(
        {
            "headers": dict(flask.request.headers),
            "time_in_hours_by_user": time_played_by_user,
        }
    )


@app.route("/get_world/<apk_version>/<user_id>/", methods=["GET"])
def get_world(apk_version: str, user_id: UserID):
    with lock:
        state = get_current_server_state(ROOT_PATH)
        world_id = state.get_world_for_user(user_id=user_id)
        last_world_id = state.get_most_recently_visited_world_by_user(user_id=user_id)
        if world_id:
            world_metadata = read_world_metadata(apk_version, world_id)
        else:
            world_metadata = None
    return flask.jsonify(
        {
            "headers": dict(flask.request.headers),
            "world_id": world_id,
            "last_world_id": last_world_id,
            "world_metadata": world_metadata,
        }
    )


@app.route("/get_worlds_user_started/<user_id>/", methods=["GET"])
def get_worlds_user_started(user_id: UserID):
    with lock:
        state = get_current_server_state(ROOT_PATH)
        worlds = state.get_world_ids_by_user_id(user_id=user_id)
    return flask.jsonify({"headers": dict(flask.request.headers), "worlds": list(worlds)})


@app.route("/get_last_world/<user_id>/", methods=["GET"])
def get_last_world(user_id: UserID):
    with lock:
        state = get_current_server_state(ROOT_PATH)
        world_id = state.get_most_recently_visited_world_by_user(user_id=user_id)
    return flask.jsonify({"headers": dict(flask.request.headers), "world_id": world_id})


@app.route("/start_world/<world_id>/<user_id>/<run_id>/", methods=["GET"])
def start_world(world_id: WorldID, user_id: UserID, run_id: RunID):
    with lock:
        state = get_current_server_state(ROOT_PATH)
        state.record_user_start_on_world(world_id=world_id, user_id=user_id, run_id=run_id)
    return flask.jsonify(
        {
            "headers": dict(flask.request.headers),
        }
    )


@app.route("/reset_world/<world_id>/<user_id>/<run_id>/", methods=["GET"])
def reset_world(world_id: WorldID, user_id: UserID, run_id: RunID):
    with lock:
        state = get_current_server_state(ROOT_PATH)
        state.record_user_reset_on_world(world_id=world_id, user_id=user_id, run_id=run_id)
    return flask.jsonify(
        {
            "headers": dict(flask.request.headers),
        }
    )


@app.route("/upload/<apk_version>/<world_id>/<user_id>/<run_id>/<filename>", methods=["POST"])
def upload(apk_version: str, world_id: WorldID, user_id: UserID, run_id: RunID, filename: str):
    data = flask.request.get_data()
    with lock:
        state = get_current_server_state(ROOT_PATH)
        state.save_user_data_for_world(
            data=data, filename=filename, apk_version=apk_version, world_id=world_id, user_id=user_id, run_id=run_id
        )
    return flask.jsonify(
        {
            "world_id": str(world_id),
            "upload_size": len(data),
        }
    )


@app.route("/download/<apk_version>/<world_id>", methods=["GET"])
def download(apk_version: str, world_id: WorldID):
    world_data = (WORLD_ROOT / apk_version / (world_id + ".zip")).read_bytes()
    world_hash = hashlib.md5(world_data).hexdigest()
    response = flask.make_response(world_data)
    response.headers["X-MD5"] = world_hash
    return response


@app.route("/apk/<participant_id>.apk")
def download_apk(participant_id: str):
    return send_from_directory(APK_DIR, f"avalon__{participant_id}.apk")


if __name__ == "__main__":
    app.config["MAX_CONTENT_LENGTH"] = 128 * 1024 * 1024
    app.run(host="0.0.0.0", port=64080)
