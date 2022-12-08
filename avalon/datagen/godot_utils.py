import json
import os
import uuid
from pathlib import Path

import boto3

from avalon.common.imports import tqdm
from avalon.common.log_utils import logger
from avalon.contrib.s3_utils import SimpleS3Client
from avalon.contrib.s3_utils import download_tar_from_s3_and_unpack
from avalon.contrib.utils import FILESYSTEM_ROOT
from avalon.contrib.utils import run_local_command
from avalon.datagen.godot_env.replay import GodotEnvReplay

GODOT_TEXTURES_PATH = Path(".") / "datagen" / "godot" / "Textures"

S3_AVALON_ERROR_BUCKET = "avalon-error-logs"


def download_godot_textures(download_path: Path = GODOT_TEXTURES_PATH) -> None:
    logger.info(f"Copying Godot textures from S3 to {download_path}")
    if not os.path.exists(str(download_path)):
        os.mkdir(download_path)

    # Create a new session to make client creation thread-safe.
    s3 = boto3.session.Session().client("s3")
    BUCKET_NAME = "godot-textures"
    bucket = boto3.resource("s3").Bucket(BUCKET_NAME)
    for s3_object in tqdm(list(bucket.objects.all())):
        path, filename = os.path.split(s3_object.key)
        filepath = download_path / filename
        if not filepath.exists():
            with open(filepath, "wb") as f:
                s3.download_fileobj(BUCKET_NAME, s3_object.key, f)


def fix_resource_paths_for_world(worlds_path: Path, target_path: Path, path_to_replace: str):
    EXCEPTED_ERRORS = {b"sed: no input files\n"}
    command = f"cd {worlds_path} && grep -rl {path_to_replace} | xargs sed -i 's+{path_to_replace}+{target_path}+g'"
    output = run_local_command(command, is_checked=False, trace_output=False)
    if output.returncode != 0 and output.stdout not in EXCEPTED_ERRORS:
        raise Exception(f"{output.stdout.decode('UTF-8')}")


def create_env_from_s3_artifacts(
    s3_key: str,
    bucket_name: str = S3_AVALON_ERROR_BUCKET,
    world_index: int = -1,
) -> GodotEnvReplay:
    s3_client = SimpleS3Client(bucket_name=bucket_name)
    output_path = Path(f"{FILESYSTEM_ROOT}/tmp/env_replay_testing/{uuid.uuid4()}/{s3_key}")
    output_path.parent.mkdir(parents=True)
    unpacked_artifacts_path = download_tar_from_s3_and_unpack(output_path, s3_client)
    return create_env_from_artifacts(unpacked_artifacts_path, world_index)


def create_env_from_artifacts(unpacked_artifacts_path: Path, world_index: int = -1) -> GodotEnvReplay:
    metadata_path = unpacked_artifacts_path / "meta.json"
    metadata = json.load(open(metadata_path))

    world_paths = unpacked_artifacts_path / "worlds"

    # note: per the convention in godot env, the world at index zero in `recent_worlds` is the oldest world
    #   to get the most recent world use `recent_worlds[-1]`
    recent_worlds = []

    for generation_id, generated_root_path in metadata["generated_root_path_by_id"].items():
        world_path = world_paths / generation_id
        fix_resource_paths_for_world(world_path, world_path, generated_root_path)
        recent_worlds.append(world_path / "main.tscn")

    return GodotEnvReplay.from_local_files(
        run_uuid=metadata["run_uuid"],
        action_path=unpacked_artifacts_path / "actions.out",
        config_path=unpacked_artifacts_path / "config.json",
        world_path=recent_worlds[world_index],
    )
