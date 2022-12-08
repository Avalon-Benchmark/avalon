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
"""The environment class for MonoBeast."""
import gym.spaces.utils
import torch


def _format_frame(frame):
    frame = torch.from_numpy(frame)
    return frame.view((1, 1) + frame.shape)  # (...) -> (T,B,...).


class Environment:
    def __init__(self, gym_env: gym.Env, device: str = "cuda:0") -> None:
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None
        self.device = device

        last_action_sample = gym_env.action_space.sample()
        self.default_action = torch.from_numpy(
            gym.spaces.utils.flatten(gym_env.action_space, last_action_sample)
        ).unsqueeze(0)

    def initial(self):
        initial_reward = torch.zeros(1, 1).to(self.device)
        # This supports only single-tensor actions ATM.

        # initial_last_action = torch.zeros(1, self.flat_action_dim, dtype=torch.float32)
        # initial_last_action[0, 0] = 1  # TODO unyuck
        initial_last_action = self.default_action.clone().to(self.device)

        self.episode_return = torch.zeros(1, 1).to(self.device)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32).to(self.device)
        initial_done = torch.ones(1, 1, dtype=torch.uint8).to(self.device)
        initial_frame = _format_frame(self.gym_env.reset()).to(self.device)
        return dict(
            frame=initial_frame,
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=initial_last_action,
        )

    def step(self, action):
        frame, reward, done, info = self.gym_env.step(action)
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if done:
            frame = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1).to(self.device)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32).to(self.device)

        frame = _format_frame(frame).to(self.device)
        reward = torch.tensor(reward).view(1, 1).to(self.device)
        done = torch.tensor(done).view(1, 1).to(self.device)

        return dict(
            frame=frame,
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step=episode_step,
            last_action=action,
            info=info,
        )

    def close(self) -> None:
        self.gym_env.close()
