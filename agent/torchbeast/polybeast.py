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
import subprocess

import numpy as np

from agent.torchbeast import polybeast_env
from agent.torchbeast import polybeast_learner
from agent.torchbeast.avalon_helpers import _destroy_process_tree
from common.log_utils import logger


def run_env(flags, actor_id, exit_event):
    np.random.seed()  # Get new random seed in forked process.
    polybeast_env.main(flags, exit_event)


def run_learner(flags):
    polybeast_learner.main(flags)


def get_flags(check_argv_empty: bool = False):
    flags = argparse.Namespace()
    flags, argv = polybeast_learner.parser.parse_known_args(namespace=flags)
    flags, argv = polybeast_env.parser.parse_known_args(args=argv, namespace=flags)
    if check_argv_empty and argv:
        # Produce an error message.
        polybeast_learner.parser.print_usage()
        print("")
        polybeast_env.parser.print_usage()
        print("Unkown args:", " ".join(argv))
        return -1
    return flags


def main(flags):
    data_path = "/mnt/private/data/level_gen"
    os.makedirs(data_path, exist_ok=True)
    shutil.rmtree(data_path)
    env_processes = []
    exit_event = mp.Event()
    for actor_id in range(1):
        p = mp.Process(target=run_env, args=(flags, actor_id, exit_event))
        p.start()
        env_processes.append(p)

    test_data = run_learner(flags)

    logger.info("Learner finished, killing processes")
    exit_event.set()

    for p in env_processes:
        _destroy_process_tree(p.pid)

    return test_data


if __name__ == "__main__":
    flags = get_flags(check_argv_empty=True)
    main(flags)
