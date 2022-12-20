# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import multiprocessing as mp
import os.path
import shutil
import tarfile
from pathlib import Path
from uuid import uuid4

import numpy as np
import wandb

from avalon.agent.evaluation import EVAL_TEMP_PATH
from avalon.agent.godot.godot_gym import CURRICULUM_BASE_PATH
from avalon.agent.torchbeast import polybeast_env
from avalon.agent.torchbeast import polybeast_learner
from avalon.agent.torchbeast.avalon_helpers import TORCHBEAST_ENV_LOGS_PATH
from avalon.agent.torchbeast.avalon_helpers import _destroy_process_tree
from avalon.common.log_utils import configure_parent_logging
from avalon.common.log_utils import logger
from avalon.contrib.s3_utils import SimpleS3Client
from avalon.contrib.utils import FILESYSTEM_ROOT


def run_env(flags, actor_id, exit_event) -> None:
    configure_parent_logging()
    np.random.seed()  # Get new random seed in forked process.
    polybeast_env.main(flags, exit_event)


def run_learner(flags):
    return polybeast_learner.main(flags)


def get_flags(check_argv_empty: bool = False):
    flags = argparse.Namespace()
    flags, argv = polybeast_learner.parser.parse_known_args(namespace=flags)
    flags, argv = polybeast_env.parser.parse_known_args(args=argv, namespace=flags)
    if check_argv_empty and argv:
        # Produce an error message.
        polybeast_learner.parser.print_usage()
        logger.info("")
        polybeast_env.parser.print_usage()
        logger.info("Unknown args:", " ".join(argv))
        return -1
    return flags


def restore_curriculum_files() -> None:
    api = wandb.Api()
    run_api = api.run(wandb.run.path)
    curriculum_filenames = [file.name for file in run_api.files() if file.name.startswith("curriculum")]
    os.makedirs(CURRICULUM_BASE_PATH, exist_ok=True)
    shutil.rmtree(CURRICULUM_BASE_PATH)
    for filename in curriculum_filenames:
        wandb.restore(filename, root=CURRICULUM_BASE_PATH.parent, replace=True)


def main(flags):
    if flags.fixed_world_key and flags.fixed_world_path:
        logger.info(f"Extracting {flags.fixed_world_key} to {flags.fixed_world_path}")
        s3_client = SimpleS3Client()
        os.makedirs(flags.fixed_world_path, exist_ok=True)
        os.makedirs(EVAL_TEMP_PATH, exist_ok=True)
        s3_client.download_to_file(flags.fixed_world_key, Path(EVAL_TEMP_PATH) / flags.fixed_world_key)
        with tarfile.open(Path(EVAL_TEMP_PATH) / flags.fixed_world_key, "r:gz") as f:
            f.extractall(flags.fixed_world_path)
    else:
        data_path = f"{FILESYSTEM_ROOT}/data/level_gen"
        os.makedirs(data_path, exist_ok=True)
        shutil.rmtree(data_path)

    os.makedirs(TORCHBEAST_ENV_LOGS_PATH, exist_ok=True)
    shutil.rmtree(TORCHBEAST_ENV_LOGS_PATH)

    if not flags.xpid:
        flags.xpid = str(uuid4())
    name = os.environ.get("EXPERIMENT_NAME", flags.env)

    if flags.mode == "train":
        wandb.init(
            config=flags,
            project=flags.project,
            name=name + f"__{flags.training_protocol}",
            id=flags.xpid,
            resume="allow",
        )
        restore_curriculum_files()
    else:
        wandb.init(
            config=flags, project=flags.project, name=name + f"__{flags.mode}", notes=f"{flags.project}/{flags.xpid}"
        )

    env_processes = []
    exit_event = mp.Event()
    for actor_id in range(1):
        p = mp.Process(target=run_env, args=(flags, actor_id, exit_event))
        p.start()
        env_processes.append(p)

    result = run_learner(flags)

    logger.info("Learner finished, killing processes")
    exit_event.set()

    for p in env_processes:
        _destroy_process_tree(p.pid)

    return result


if __name__ == "__main__":
    flags = get_flags(check_argv_empty=True)
    main(flags)
