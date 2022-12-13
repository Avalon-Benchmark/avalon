from __future__ import annotations

from collections import defaultdict
from typing import Tuple
from typing import Type

import attr
import gym
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from tree import map_structure

from avalon.agent.common import wandb_lib
from avalon.agent.common.action_model import DictActionDist
from avalon.agent.common.action_model import visualize_action_dists
from avalon.agent.common.types import ActionBatch
from avalon.agent.common.types import Algorithm
from avalon.agent.common.types import ObservationBatch
from avalon.agent.common.util import explained_variance
from avalon.agent.ppo.gae import gae
from avalon.agent.ppo.model import CNNBase
from avalon.agent.ppo.model import MLPBase
from avalon.agent.ppo.observation_model import ObservationModel
from avalon.agent.ppo.observation_model import PPOModel
from avalon.agent.ppo.params import PPOParams
from avalon.agent.ppo.ppo_types import PPOBatchDataWithGAE
from avalon.agent.ppo.ppo_types import PPOBatchSequenceDataWithGAE
from avalon.agent.ppo.ppo_types import PPOStepData
from avalon.contrib.utils import set_all_seeds


class PPO(Algorithm["PPOParams"]):
    step_data_type: Type = PPOStepData

    def __init__(self, params: PPOParams, observation_space: gym.spaces.Dict, action_space: gym.spaces.Dict) -> None:
        self.params = params
        super().__init__(params, observation_space, action_space)

        self.model: PPOModel
        if params.model == "cnn":
            self.model = CNNBase(params, observation_space, action_space, hidden_size=64)
        elif params.model == "mlp":
            self.model = MLPBase(params, observation_space, action_space)
        elif params.model == "godot":
            self.model = ObservationModel(params, observation_space, action_space)
        else:
            assert False

        self.optim = torch.optim.Adam(self.model.parameters(), lr=params.lr, eps=1e-5)
        self.seed_state = defaultdict(int)

    def forward(self, obs: ObservationBatch):
        return self.model(obs)

    def rollout_step(
        self,
        next_obs: ObservationBatch,
        dones: Tensor,  # shape (batch_size, )
        indices_to_run: list[bool],  # shape (batch_size, )
        exploration_mode: str,
    ) -> Tuple[ActionBatch, dict]:
        step_values: Tensor
        dist: DictActionDist
        step_values, dist = self.model(next_obs)

        # Sample actions from the policy distribution
        # This should be a list (of len num_workers) of action dicts
        step_actions = dist.sample()

        if self.params.deterministic:
            # Determinism here is tricky, since we allow partial batches.
            # Need to do a separate sample for each worker, setting the seed before each,
            # otherwise the random "seed" for each sample will differ depending on ordering.
            # This will overwrite the previous action sample, to avoid having to construct the arrays.
            for i, should_run in enumerate(indices_to_run):
                # We do need a new seed for each sample,
                # otherwise we always sample the same action as long as the probs are roughly uniform
                # (it's not a chaotic sampling algorithm).
                # So we construct one based on worker_id that gets incremented in a determinstic way
                # irrelevant of the ordering of partial batches.
                if should_run:
                    if self.seed_state[i] == 0:
                        self.seed_state[i] = i * 100000
                    set_all_seeds(self.seed_state[i])

                    def assign(target, source) -> None:
                        target[i] = source

                    map_structure(assign, step_actions, dist[i].sample())
                    self.seed_state[i] += 1

        step_policy_probs = dist.log_prob(step_actions)
        policy_entropy = dist.entropy().cpu()

        # Store data for use in training
        step_actions = map_structure(lambda x: x.detach().cpu(), step_actions)

        step_values = step_values.detach().cpu()
        step_policy_probs = step_policy_probs.detach().cpu()
        to_store = {"value": step_values, "policy_prob": step_policy_probs, "policy_entropy": policy_entropy}
        return step_actions, to_store

    def train_step(self, rollouts: PPOBatchSequenceData, i: int) -> int:  # type: ignore
        # the observations have an extra frame at the end to use for value backup.
        final_observations: ObservationBatch = {k: v[:, -1] for k, v in rollouts.observation.items()}
        rollouts = attr.evolve(rollouts, observation=map_structure(lambda x: x[:, :-1], rollouts.observation))

        if self.params.entropy_mode == "max" and self.params.entropy_coef != 0:
            entropy_reward: Tensor = rollouts.policy_entropy * self.params.entropy_coef
            wandb_lib.log_histogram("training/entropy_reward", entropy_reward, i)
            wandb_lib.log_histogram("training/env_reward", rollouts.reward, i)
            rollouts = attr.evolve(rollouts, reward=rollouts.reward + entropy_reward)

        with torch.no_grad():
            final_values = self.model(final_observations)[0].detach()

            # Compute advantages and future discounted rewards with GAE
            advantages = gae(
                rollouts.reward, rollouts.value, rollouts.done, self.params.discount, self.params.lam, final_values
            )
            rewards_to_go = advantages + rollouts.value
            rollouts = PPOBatchSequenceDataWithGAE(
                **attr.asdict(rollouts), advantage=advantages, reward_to_go=rewards_to_go
            )

            # flatten (num_workers,num_steps,...) into ((num_workers*num_steps,...)
            train_data: PPOBatchDataWithGAE = map_structure(lambda x: rearrange(x, "b t ... -> (b t) ..."), rollouts)

        for epoch in range(self.params.ppo_epochs):
            indices = np.random.permutation(self.params.num_steps * self.params.num_workers)

            for batch in range(self.params.num_batches):
                start = batch * self.params.batch_size
                end = start + self.params.batch_size

                batch_indices = indices[start:end]
                batch_data = map_structure(lambda x: x[batch_indices], train_data)
                self.train_substep(batch_data, i)

                i += 1
            assert end == self.params.num_steps * self.params.num_workers

        # Lots of logging!
        ev = explained_variance(rollouts.value.detach().flatten().cpu(), rewards_to_go.detach().flatten().cpu())
        wandb_lib.log_scalar("training/rollout_value_ev", ev, i)
        wandb_lib.log_histogram("training/raw_advantage", advantages, i)
        wandb_lib.log_histogram("training/rewards", rollouts.reward, i)

        return i

    def train_substep(self, batch: PPOBatchDataWithGAE, i: int) -> None:
        self.model.train()
        self.optim.zero_grad()

        # Normalize the advantages.
        # Note the advantages are normalized by batch, not the entire epoch altogether
        # might want to parameterize the constant here. I had it at 1e-5 originally
        advantages = (batch.advantage - batch.advantage.mean()) / (batch.advantage.std() + 1e-8)

        values_new, dist_new = self.model(batch.observation)
        selected_prob_new = dist_new.log_prob(batch.action)

        # Compute the PPO loss
        prob_ratio = torch.exp(selected_prob_new - batch.policy_prob)
        a = prob_ratio * advantages
        b = torch.clamp(prob_ratio, 1 - self.params.clip_range, 1 + self.params.clip_range) * advantages
        ppo_loss = -1 * torch.min(a, b)
        ppo_loss = ppo_loss.mean()

        # Compute the value function loss
        # Clipped loss - same idea as PPO loss, don't allow value to move too
        # far from where it was previously
        if self.params.clip_range_vf is None:
            value_pred_clipped = values_new
        else:
            value_pred_clipped = batch.value + (values_new - batch.value).clamp(
                -self.params.clip_range_vf, self.params.clip_range_vf
            )

        # NOTE: implementations differ in how they handle this.
        if self.params.baselines_style_vf_loss:
            # this is how it's done in baselines ppo2
            value_losses = (values_new - batch.reward_to_go) ** 2
            value_losses_clipped = (value_pred_clipped - batch.reward_to_go) ** 2
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
            value_loss = value_loss.mean()
        else:
            # but sb3 uses a simpler method:
            value_loss = F.mse_loss(batch.reward_to_go, value_pred_clipped)
        policy_entropy = torch.mean(dist_new.entropy())

        base_loss = ppo_loss + self.params.value_loss_coef * value_loss
        if self.params.entropy_mode == "regularized" and self.params.entropy_coef != 0:
            entropy_loss = -1 * self.params.entropy_coef * policy_entropy
            loss = base_loss + entropy_loss
            wandb_lib.log_scalar("loss/entropy_loss", entropy_loss, i)
        else:
            loss = base_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_grad_norm)
        self.optim.step()

        # Lots of logging
        clip_fraction = torch.mean((torch.abs(prob_ratio - 1) > self.params.clip_range).float()).item()
        wandb_lib.log_scalar("loss/clip_fraction", clip_fraction, i)
        wandb_lib.log_histogram("loss/ppo_loss", -1 * torch.min(a, b), i)
        wandb_lib.log_scalar("loss/value_loss", value_loss, i)
        wandb_lib.log_scalar("loss/loss", loss, i)
        wandb_lib.log_histogram("training/value_pred", values_new, i)
        wandb_lib.log_histogram("training/advantage", advantages, i)
        wandb_lib.log_histogram("training/value_target", batch.reward_to_go, i)
        wandb_lib.log_histogram("training/prob_ratio", prob_ratio, i)
        wandb_lib.log_scalar("policy/policy entropy", policy_entropy, i)
        # Approximate KL divergence between the policy when it was rolled out and now.
        # Explained in https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ and
        # http://joschu.net/blog/kl-approx.html
        approxkl = 0.5 * ((batch.policy_prob - selected_prob_new) ** 2).mean()
        wandb_lib.log_scalar("policy/approx_kl", approxkl, i)

        # Log actions
        visualize_action_dists(self.action_space, dist_new, prefix="train_policy")
