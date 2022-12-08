import filecmp
import gzip
import os
import shutil
import tarfile
from pathlib import Path
from typing import IO
from typing import cast

import numpy as np
from godot_parser import Node as GDNode
from loguru import logger

from avalon.common.utils import TEMP_DIR
from avalon.contrib.s3_utils import SimpleS3Client
from avalon.contrib.utils import create_temp_file_path
from avalon.datagen.errors import ImpossibleWorldError
from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.export import get_eval_agent_export_config
from avalon.datagen.world_creation.configs.export import get_oculus_export_config
from avalon.datagen.world_creation.constants import AvalonTask
from avalon.datagen.world_creation.indoor.builders import DefaultEntranceBuilder
from avalon.datagen.world_creation.indoor.building import Building
from avalon.datagen.world_creation.indoor.building import BuildingTask
from avalon.datagen.world_creation.indoor.task_generator import create_building_for_skill_scenario
from avalon.datagen.world_creation.indoor.task_generator import make_indoor_task_world
from avalon.datagen.world_creation.types import GodotScene
from avalon.datagen.world_creation.types import Point3DNP
from avalon.datagen.world_creation.world_generator import GENERATION_FUNCTION_BY_TASK
from avalon.datagen.world_creation.worlds.compositional import get_building_task_generator_class
from avalon.datagen.world_creation.worlds.export import export_world


def export_building(building: Building, building_name: str, export_path: Path) -> None:
    scene = GodotScene()
    root_node = GDNode("root", "Spatial")
    with scene.use_tree() as tree:
        tree.root = root_node
        building.export(scene, root_node, building_name)
    scene.write(str(export_path.absolute()))


def reset_unix_epoch_mtime(tar_info: tarfile.TarInfo) -> tarfile.TarInfo:
    tar_info.uid = tar_info.gid = 0
    tar_info.uname = tar_info.gname = "user"
    tar_info.mtime = 1
    return tar_info


def make_tarball(root_path: Path, glob_pattern: str, archive_path: Path) -> None:
    last_pwd = os.getcwd()
    try:
        with cast(IO[bytes], gzip.GzipFile(archive_path, mode="wb", mtime=1)) as gzip_file:
            with tarfile.open("a.tar", "w", fileobj=gzip_file) as tarball:
                os.chdir(root_path)
                for entry in root_path.glob(glob_pattern):
                    entry_path = entry.relative_to(root_path)
                    tarball.add(entry_path, filter=reset_unix_epoch_mtime)
    finally:
        os.chdir(last_pwd)


def create_indoor_world(
    rand: np.random.Generator,
    difficulty: float,
    building_task: BuildingTask,
    radius: float,
    location: Point3DNP,
    export_path: Path,
    export_config: ExportConfig = get_oculus_export_config(),
) -> None:
    task_generator = get_building_task_generator_class(building_task)()
    building, entities, spawn_location, target_location = create_building_for_skill_scenario(
        rand,
        difficulty,
        task_generator,
        location,
        site_radius=radius,
        is_indoor_only=True,
    )
    with export_config.mutable_clone() as final_export_config:
        final_export_config.is_tiled = False
    export_config = final_export_config
    world = make_indoor_task_world(
        building, entities, difficulty, spawn_location, target_location, rand, final_export_config
    )
    export_world(export_path, rand, world)


def add_entrance_if_has_none(building: Building, rand: np.random.Generator) -> None:
    if not any([len(story.entrances) > 0 for story in building.stories]):
        entrance_builder = DefaultEntranceBuilder(top=False, bottom=True)
        entrance_builder.add_entrance(building.stories[0], rand)
        assert len(building.stories[0].entrances) == 1


def get_reference_data_path(reference_name: str, snapshot_commit: str) -> Path:
    qualified_reference_name = f"{reference_name}_reference"
    cached_reference_path = Path(TEMP_DIR) / f"{qualified_reference_name}__{snapshot_commit}"
    if not cached_reference_path.exists():
        s3_client = SimpleS3Client()
        archive_path = cached_reference_path.parent / f"{cached_reference_path.name}.tar.gz"
        logger.info(f"Reference {reference_name} not found locally, fetching from S3...")
        s3_client.download_to_file(f"{snapshot_commit}/{qualified_reference_name}.tar.gz", archive_path)
        cached_reference_path.mkdir()
        shutil.unpack_archive(archive_path, cached_reference_path)
    return cached_reference_path


def compare_files(generated_file_path: Path, reference_file_path: Path) -> None:
    # Comparing the file contents directly generates a diff too big for Pycharm to view directly; instead we
    # store a copy of the files and output the paths, which you can throw into the diff tool yourself
    if not filecmp.cmp(generated_file_path, reference_file_path):
        with create_temp_file_path(cleanup=False) as generated_file_copy_path, create_temp_file_path(
            cleanup=False
        ) as reference_file_copy_path:
            shutil.copyfile(generated_file_path, generated_file_copy_path)
            shutil.copyfile(reference_file_path, reference_file_copy_path)
            assert False, (
                f"Generated and reference files ({generated_file_path.name}) don't match\n"
                f"Generated: {generated_file_copy_path}\nReference: {reference_file_copy_path}"
            )


def create_world(
    task: AvalonTask,
    difficulty: float,
    seed: int,
    export_path: Path,
    export_config: ExportConfig = get_eval_agent_export_config(),
) -> Path:
    # todo: used commonly - generalize, push up, replace other calls?
    rand = np.random.default_rng(seed)
    generation_function = GENERATION_FUNCTION_BY_TASK[task]
    try:
        generation_function(rand, difficulty, export_path, export_config)
    except ImpossibleWorldError:
        logger.error(
            f"Got ImpossibleWorldError while generating world for {task=}, {difficulty=}, {seed=}, path will be empty"
        )
    except Exception as e:
        # logger.error(f"Got '{type(e)}: {e.args}' while generating world for {task=}, {difficulty=}, {seed=}")
        raise
    return export_path


GENERIC_PATH_PLACEHOLDER = "<replaced for reproducibility>"


def make_world_file_generic(path: Path) -> None:
    with open(path, "r+") as file:
        data = file.read()
        file.seek(0)
        file.truncate()
        file.write(data.replace(str(path.parent), GENERIC_PATH_PLACEHOLDER))
