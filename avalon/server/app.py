#!/usr/bin/env python3
import hashlib
import multiprocessing
import os
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Dict

import attr
import flask
import sentry_sdk
from flask import send_from_directory
from flask import url_for
from sentry_sdk.integrations.flask import FlaskIntegration
from werkzeug.utils import secure_filename

# note: using relative imports because this server runs independent from the rest of standalone/avalon
from .server_state import RunID
from .server_state import ServerState
from .server_state import UserID
from .server_state import WorldID
from .server_state import get_active_time_played_by_users
from .server_state import get_current_server_state
from .server_state import get_world_root_path
from .server_state import read_world_metadata

IS_SENTRY_ENABLED = True

API_VERSION: str = os.getenv("API_VERSION")
assert API_VERSION is not None
ENV_ROOT_PATH = os.getenv("ROOT_PATH", None)
assert ENV_ROOT_PATH is not None

ROOT_PATH = Path(ENV_ROOT_PATH) / API_VERSION
APK_DIR = Path(ENV_ROOT_PATH) / "apks" / API_VERSION
APK_DIR.mkdir(parents=True, exist_ok=True)

if IS_SENTRY_ENABLED:
    # Uses the SENTRY_DSN environment variable. No events are sent if the variable is not set.
    sentry_sdk.init(
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


@app.route("/add_apk_version/<apk_version>/", methods=["GET"])
def add_apk_version(apk_version: str):
    with lock:
        state = get_current_server_state(ROOT_PATH, API_VERSION)
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
        state = get_current_server_state(ROOT_PATH, API_VERSION)
    return flask.jsonify(
        {
            "headers": dict(flask.request.headers),
            "apk_versions": list(state.valid_apk_versions),
        }
    )


@app.route("/verify/<apk_version>/", methods=["GET"])
def verify(apk_version: str):
    with lock:
        state = get_current_server_state(ROOT_PATH, API_VERSION)
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
        state = get_current_server_state(ROOT_PATH, API_VERSION)
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
        state = get_current_server_state(ROOT_PATH, API_VERSION)
        user_ids = list(state.get_all_user_ids())
        with open("participant_ids.txt", "r") as f:
            all_user_ids = set(f.read().strip().split("\n"))
    return flask.jsonify(
        {
            "headers": dict(flask.request.headers),
            "active_user_ids": user_ids,
            "bad_user_ids": list(all_user_ids - set(user_ids)),
        }
    )


@app.route("/time_played_by_user/", methods=["GET"])
def time_played_by_user():
    with lock:
        state = get_current_server_state(ROOT_PATH, API_VERSION)
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
        state = get_current_server_state(ROOT_PATH, API_VERSION)
        world_id = state.get_world_for_user(user_id=user_id)
        last_world_id = state.get_most_recently_visited_world_by_user(user_id=user_id)
        if world_id:
            world_metadata = read_world_metadata(
                world_root=get_world_root_path(ROOT_PATH),
                apk_version=apk_version,
                world_id=world_id,
            )
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
        state = get_current_server_state(ROOT_PATH, API_VERSION)
        worlds = state.get_world_ids_by_user_id(user_id=user_id)
    return flask.jsonify({"headers": dict(flask.request.headers), "worlds": list(worlds)})


@app.route("/get_last_world/<user_id>/", methods=["GET"])
def get_last_world(user_id: UserID):
    with lock:
        state = get_current_server_state(ROOT_PATH, API_VERSION)
        world_id = state.get_most_recently_visited_world_by_user(user_id=user_id)
    return flask.jsonify({"headers": dict(flask.request.headers), "world_id": world_id})


@app.route("/start_world/<world_id>/<user_id>/<run_id>/", methods=["GET"])
def start_world(world_id: WorldID, user_id: UserID, run_id: RunID):
    with lock:
        state = get_current_server_state(ROOT_PATH, API_VERSION)
        state.record_user_start_on_world(world_id=world_id, user_id=user_id, run_id=run_id)
    return flask.jsonify(
        {
            "headers": dict(flask.request.headers),
        }
    )


@app.route("/reset_world/<world_id>/<user_id>/<run_id>/", methods=["GET"])
def reset_world(world_id: WorldID, user_id: UserID, run_id: RunID):
    with lock:
        state = get_current_server_state(ROOT_PATH, API_VERSION)
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
        state = get_current_server_state(ROOT_PATH, API_VERSION)
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
    world_data = (get_world_root_path(root_path=ROOT_PATH) / apk_version / (world_id + ".zip")).read_bytes()
    world_hash = hashlib.md5(world_data).hexdigest()
    response = flask.make_response(world_data)
    response.headers["X-MD5"] = world_hash
    return response


@app.route("/download_apk/<participant_id>/")
def download_apk(participant_id: str):
    return send_from_directory(APK_DIR, f"avalon__{participant_id}.apk")


@app.route("/upload_apk/", methods=["POST"])
def upload_apk():
    file = flask.request.files["file"]
    filename = secure_filename(file.filename)
    file.save(os.path.join(APK_DIR, filename))
    return flask.jsonify({"download_url": url_for("download_apk", participant_id=filename.split(".")[0])})


def get_stats(state: ServerState) -> Dict[str, Any]:
    total_visited_worlds = 0
    for user_ids in state.user_ids_by_world_id.values():
        if len(user_ids) > 0:
            total_visited_worlds += 1

    ignored_world_id_by_task = defaultdict(set)

    for world_id in state.ignored_world_ids:
        task_name = world_id.split("__")[0]
        ignored_world_id_by_task[task_name].add(world_id)

    impossible_worlds = {}
    for world_id, reset_user_ids in state.resets_by_world_id.items():
        all_user_ids_on_world = state.user_ids_by_world_id[world_id]
        plays_with_reset = len(all_user_ids_on_world) - len(reset_user_ids)
        if world_id not in state.ignored_world_ids and (
            (len(reset_user_ids) >= 3 and plays_with_reset < 5)
            or (state.is_successful_play_on_world(world_id) and plays_with_reset > 5)
        ):
            impossible_worlds[world_id] = {
                "num_resets": len(reset_user_ids),
                "num_plays": len(all_user_ids_on_world),
                "is_successful_play_on_world": state.is_successful_play_on_world(world_id),
            }

    return {
        "total_visited_worlds": total_visited_worlds,
        "ignored_world_id_by_task": {k: list(v) for k, v in ignored_world_id_by_task.items()},
        "impossible_worlds": impossible_worlds,
    }


@app.route("/stats/", methods=["GET"])
def stats():
    with lock:
        state = get_current_server_state(ROOT_PATH, API_VERSION)
        return flask.jsonify(
            {
                "headers": dict(flask.request.headers),
                "stats": get_stats(state),
            }
        )


if __name__ == "__main__":
    app.config["MAX_CONTENT_LENGTH"] = 3 * 1024 * 1024 * 1024
    app.run(host="0.0.0.0", port=64080)
