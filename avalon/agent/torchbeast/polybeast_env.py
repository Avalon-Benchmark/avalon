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
import logging
import multiprocessing as mp
import time

import libtorchbeast
import numpy as np

# yapf: disable
from loguru import logger

from avalon.agent.torchbeast import atari_wrappers
from avalon.agent.torchbeast import avalon_helpers
from avalon.agent.torchbeast.avalon_helpers import create_godot_env
from avalon.agent.torchbeast.avalon_helpers import godot_config_from_flags
from avalon.common.log_utils import configure_parent_logging

parser = argparse.ArgumentParser(description='Remote Environment Server')

parser.add_argument("--pipes_basename", default="unix:/tmp/polybeast",
                    help="Basename for the pipes for inter-process communication. "
                    "Has to be of the type unix:/some/path.")
parser.add_argument('--num_servers', default=8, type=int, metavar='N',
                    help='Number of environment servers.')
parser.add_argument('--env', type=str, default='godot',
                    help='Gym environment.')
parser.add_argument('--env_seed', default=0, type=int,
                    help='Seed for environment.')
parser.add_argument('--task_difficulty_update', default=5e-3, type=float,
                    help='Update rate for env task difficulty.')
parser.add_argument('--meta_difficulty_update', default=2e-4, type=float,
                    help='Update rate for env meta difficulty.')
parser.add_argument('--energy_cost_coefficient', default=1e-8, type=float,
                    help='Coefficient for energy usage')
parser.add_argument('--head_pitch_coefficient', default=0, type=float,
                    help='Coefficient for body kinetic energy')
parser.add_argument('--head_roll_coefficient', default=0, type=float,
                    help='Coefficient for body kinetic energy')
parser.add_argument('--time_limit', default=120, type=int,
                    help='Time limit at 0 difficulty.')
parser.add_argument('--training_protocol', default="multi_task_basic",
                    help='Training protocol from paper')
parser.add_argument("--is_task_curriculum_disabled", action="store_true",
                    help="Is task curriculum disabled during training")
parser.add_argument("--is_meta_curriculum_enabled", action="store_true",
                    help="Is meta curriculum enabled during training")
parser.add_argument('--fixed_world_max_difficulty', default=0.5, type=float,
                    help='Fixed world max difficulty')
parser.add_argument('--fixed_world_path', type=str, default='',
                    help='Path for loading eval worlds')
parser.add_argument('--fixed_world_key', type=str, default='',
                    help='Key for loading eval worlds')
parser.add_argument('--load_checkpoint_filename', type=str, default='',
                    help='Checkpoint filename to load')

parser.add_argument("--energy_cost_aggregator", default="sum",
                    help="How to aggregate energy costs.")
# yapf: enable


class Env:
    def reset(self):
        logger.debug("reset called")
        return np.ones((4, 84, 84), dtype=np.uint8)

    def step(self, action):
        frame = np.zeros((4, 84, 84), dtype=np.uint8)
        return frame, 0.0, False, {}  # First three mandatory.


def create_env(env_name, config):
    if env_name == "godot":
        return create_godot_env(config)
    else:
        return atari_wrappers.wrap_pytorch(
            avalon_helpers.wrap_atari(
                atari_wrappers.wrap_deepmind(
                    atari_wrappers.make_atari(env_name),
                    clip_rewards=False,
                    frame_stack=True,
                    scale=False,
                )
            )
        )


def serve(env_name, server_address, config) -> None:
    configure_parent_logging()
    logging.info(f"serve called with config {config}")
    init = Env if env_name == "Mock" else lambda: create_env(env_name, config)
    server = libtorchbeast.Server(init, server_address=server_address)
    server.run()


def main(flags, exit_event: mp.Event):
    if not flags.pipes_basename.startswith("unix:"):
        raise Exception("--pipes_basename has to be of the form unix:/some/path.")

    processes = []
    for i in range(flags.num_servers):
        env_config = godot_config_from_flags(flags, env_index=i)
        p = mp.Process(target=serve, args=(flags.env, f"{flags.pipes_basename}.{i}", env_config))
        p.start()
        processes.append(p)

    try:
        # We are only here to listen to the interrupt.
        while True:
            time.sleep(1)
            if exit_event.is_set():
                logger.debug("exit event set")
                for p in processes:
                    p.kill()
                return
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
