"""
To maintain experiment reproducibility, this script is used to generate reference buildings / worlds in the exact state
that they were during data collection / experiments as published in the Avalon paper. These are then used for regression
testing to ensure refactoring / cleaning up code doesn't intentionally break reproducibility. (See test_indoor.py)

Whenever we release a new version of the environment, new references can be generated and committed.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Callable
from typing import List
from typing import Tuple

import attr
import numpy as np

from avalon.common.errors import SwitchError
from avalon.common.log_utils import logger
from avalon.contrib.s3_utils import SimpleS3Client
from avalon.contrib.utils import TEMP_DIR
from avalon.contrib.utils import is_git_repo_clean_on_notebook
from avalon.contrib.utils import temp_dir
from avalon.datagen.world_creation.tests.fixtures import ChecksumManifest
from avalon.datagen.world_creation.tests.fixtures import get_current_reference_manifest
from avalon.datagen.world_creation.tests.helpers import create_indoor_world
from avalon.datagen.world_creation.tests.helpers import create_world
from avalon.datagen.world_creation.tests.helpers import export_building
from avalon.datagen.world_creation.tests.helpers import get_reference_data_path
from avalon.datagen.world_creation.tests.helpers import make_tarball
from avalon.datagen.world_creation.tests.helpers import make_world_file_generic
from avalon.datagen.world_creation.tests.params import BUILDING_CATALOG
from avalon.datagen.world_creation.tests.params import CANONICAL_SEED
from avalon.datagen.world_creation.tests.params import INDOOR_WORLD_CATALOG
from avalon.datagen.world_creation.tests.params import OUTDOOR_WORLD_CATALOG
from avalon.datagen.world_creation.worlds.world import build_building


def generate_reference_buildings(export_path_base: Path) -> None:
    building_id = 0
    for config_id, building_config in BUILDING_CATALOG.items():
        logger.info(f"Generating reference building {config_id}")

        building_name = f"{config_id}__{CANONICAL_SEED}"
        export_path = export_path_base / f"{building_name}.tscn"

        rand = np.random.default_rng(CANONICAL_SEED)
        try:
            building = build_building(building_config, building_id, rand)
            export_building(building, building_name, export_path)
        except Exception as e:
            logger.warning(f"{config_id} errored: {e}; writing empty file")
            # For any building configs that don't pan out (incompatible or broken) we write an empty file
            export_path.write_text("")


def generate_reference_indoor_worlds(export_path_base: Path) -> None:
    for world_name, params in INDOOR_WORLD_CATALOG.items():
        logger.info(f"Generating reference indoor world {world_name}")
        output_path = export_path_base / world_name
        output_path.mkdir()
        create_indoor_world(*params, export_path=output_path)


def generate_reference_outdoor_worlds(export_path_base: Path) -> None:
    for world_name, params in OUTDOOR_WORLD_CATALOG.items():
        logger.info(f"Generating reference outdoor world {world_name}")
        output_path = export_path_base / world_name
        output_path.mkdir()
        create_world(*params, export_path=output_path)
        for file in output_path.iterdir():
            make_world_file_generic(file)


def clean_path(path: Path, glob_pattern: str) -> None:
    for entry in path.glob(glob_pattern):
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def archive_and_clean(source_path: Path, glob_pattern: str, archive_name: str) -> None:
    make_tarball(source_path, glob_pattern, source_path / archive_name)
    clean_path(source_path, glob_pattern)


def get_path_checksum(path: Path) -> str:
    md5 = hashlib.md5()
    for file in sorted(path.iterdir()):
        md5.update(file.read_bytes())
    return md5.hexdigest()


def get_file_checksum(file_path: Path) -> str:
    md5 = hashlib.md5()
    md5.update(file_path.read_bytes())
    return md5.hexdigest()


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def make_checksum_json(source_path: Path, snapshot_commit: str, json_output_path: Path) -> None:
    checksums: ChecksumManifest = dict(snapshot_commit=snapshot_commit, checksums={})
    for entry in sorted(source_path.iterdir()):
        if entry.is_dir():
            checksums["checksums"][entry.stem] = get_path_checksum(entry)
        else:
            checksums["checksums"][entry.stem] = get_file_checksum(entry)
    return json.dump(checksums, open(json_output_path, "w"), indent=2)


class Mode(Enum):
    REVIEW = "review"
    UPDATE = "update"


@attr.s(auto_attribs=True, eq=False, collect_by_mro=True)
class DataReference:
    name: str
    generate_data: Callable[[Path], None]
    glob_pattern: str


def make_reference_review(reference: DataReference, data_path: Path) -> Tuple[Path, Path]:
    review_path = data_path / "reference_review" / reference.name
    review_current_path = review_path / "current"
    review_new_path = review_path / "new"
    assert not review_path.exists(), f"Review path {review_path} already exists, remove it before generating a new one"
    review_path.mkdir(parents=True)
    review_new_path.mkdir()

    try:
        reference_manifest = get_current_reference_manifest(reference.name)
        current_reference_path = get_reference_data_path(reference.name, reference_manifest["snapshot_commit"])
    except FileNotFoundError:
        logger.warning(f"warning: No reference manifest found for {reference.name}, first time generating these?")
        current_reference_path = None
    if current_reference_path is not None:
        shutil.copytree(current_reference_path, review_current_path)
    reference.generate_data(review_new_path)
    manifest_file_name = f"{reference.name}_manifest.json"
    manifest_path = review_new_path / manifest_file_name
    make_checksum_json(review_new_path, get_git_revision_hash(), manifest_path)
    return review_current_path, review_new_path


def update_reference(reference: DataReference, commit_hash: str, data_path: Path, upload_to_s3: bool = True) -> None:
    with temp_dir(TEMP_DIR) as export_path_base:
        reference.generate_data(export_path_base)

        manifest_file_name = f"{reference.name}_manifest.json"
        manifest_path = export_path_base / manifest_file_name
        make_checksum_json(export_path_base, commit_hash, manifest_path)
        (data_path / manifest_file_name).unlink(missing_ok=True)
        shutil.move(manifest_path, data_path)

        if upload_to_s3:
            s3_client = SimpleS3Client()
            tarball_name = f"{reference.name}_reference.tar.gz"
            tarball_path = export_path_base / tarball_name
            make_tarball(export_path_base, reference.glob_pattern, tarball_path)
            logger.info(f"Uploading {reference.name} to S3...")
            s3_client.upload_from_file(tarball_path, f"{commit_hash}/{tarball_name}")


def main(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", dest="mode", required=True, choices=[mode.value for mode in Mode])
    mode = Mode(parser.parse_args(args[1:]).mode)

    if mode == Mode.UPDATE and not is_git_repo_clean_on_notebook():
        logger.warning(
            "Warning: Asked to generate references, but git repo is not clean. Are you sure you want to do this?"
        )
        commit_hash = input("If so, enter the commit hash that these references should refer to:\n").strip()
    else:
        commit_hash = get_git_revision_hash()

    data_path = Path(__file__).parent / "data"
    references = [
        DataReference("buildings", generate_reference_buildings, "*.tscn"),
        DataReference("indoor_worlds", generate_reference_indoor_worlds, "indoor_world*"),
        DataReference("outdoor_worlds", generate_reference_outdoor_worlds, "outdoor_world*"),
    ]
    for reference in references:
        if mode == Mode.REVIEW:
            current_reference_path, new_reference_path = make_reference_review(reference, data_path)
            logger.info(
                f"Generated reference review data for {reference.name}\n"
                f"\tcurrent: {current_reference_path}\n"
                f"\tnew: {new_reference_path}"
            )
        elif mode == Mode.UPDATE:
            update_reference(reference, commit_hash, data_path)
            logger.info(f"Updated reference data and manifest for {reference.name}")
        else:
            raise SwitchError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main(sys.argv)
