import gym
import numpy as np
import torch
import torch.nn.functional as F
from tree import map_structure

from agent.observation_model import ObservationModel
from agent.ppo import wandb_lib
from agent.ppo.gae import gae
from agent.ppo.model import CNNBase
from agent.ppo.model import MLPBase
from agent.wrappers import OneHotMultiDiscrete


class PPO(torch.nn.Module):
    def __init__(self, args, observation_space, action_space):
        super().__init__()
        self.args = args
        self.observation_space = observation_space
        self.action_space = action_space

        if args.model == "cnn":
            self.model = CNNBase(args, observation_space, action_space, hidden_size=64)
        elif args.model == "mlp":
            self.model = MLPBase(args, observation_space, action_space)
        elif args.model == "godot":
            self.model = ObservationModel(args, observation_space, action_space, args.model_params)
        else:
            assert False

        self.optim = torch.optim.Adam(self.model.parameters(), lr=args.lr, eps=1e-5)

    def forward(self, obs):
        return self.model(obs)

    def rollout_step(self, next_obs, dones, indices_to_run):
        next_obs = {k: v[indices_to_run] for k, v in next_obs.items()}
        step_values, dist = self.model(next_obs)
        # assert step_values.shape == (self.num_workers,)

        # Sample actions from the policy distribution
        # This should be a list (of len num_workers) of action dicts
        step_actions = dist.sample()
        step_policy_probs = dist.log_prob(step_actions)

        # Store data for use in training
        step_actions = map_structure(lambda x: x.detach().cpu().numpy(), step_actions)

        step_values = step_values.detach().cpu().numpy()
        step_policy_probs = step_policy_probs.detach().cpu().numpy()
        to_store = {"values": step_values, "policy_probs": step_policy_probs}
        return step_actions, to_store

    def train_batch(self, rollouts, final_obs, i):
        # Naming not the best; train on the entire replay buffer.
        with torch.no_grad():
            rollouts = map_structure(lambda x: torch.tensor(x).to(self.args.device), rollouts)

            final_observations = map_structure(lambda x: torch.tensor(x).to(self.args.device), final_obs)
            final_values = self.model(final_observations)[0].detach()

            # Compute advantages and future discounted rewards with GAE
            rewards = rollouts["rewards"]
            values = rollouts["values"]
            dones = rollouts["dones"]
            advantages = gae(rewards, values, dones, self.args.discount, self.args.lam, final_values)
            rewards_to_go = advantages + values
            rollouts["advantages"] = advantages
            rollouts["rewards_to_go"] = rewards_to_go

            # flatten (num_workers,num_steps,...) into ((num_workers*num_steps,...)
            train_data = map_structure(lambda x: x.reshape((-1,) + x.shape[2:]), rollouts)

        for epoch in range(self.args.ppo_epochs):
            indices = np.random.permutation(self.args.num_steps * self.args.num_workers)

            for batch in range(self.args.num_batches):
                start = batch * self.args.batch_size
                end = start + self.args.batch_size

                batch_indices = indices[start:end]
                batch_data = map_structure(lambda x: x[batch_indices], train_data)
                self.train_step(batch_data, i)

                i += 1

        # Lots of logging!
        # ev = explained_variance(
        #     values.detach().flatten().cpu().numpy(), rewards_to_go.detach().flatten().cpu().numpy()
        # )
        # wandb_lib.log_scalar("training/rollout_value_ev", ev, i)

        # for k, head in model.model.action_head.action_heads.items():
        #     if isinstance(head, NormalHead):
        #         wandb_lib.log_scalar(f"policy/actions/{k}_std", head.log_std.exp().mean(), i)

        wandb_lib.log_histogram("training/raw_advantage", advantages, i)
        wandb_lib.log_histogram("training/rewards", rewards, i)
        observations = {k.removeprefix("obs__"): v for k, v in rollouts.items() if k.startswith("obs__")}
        for k, v in self.observation_space.items():
            if len(v.shape) == 3:
                obs_video = observations[k][:8]
                if "godot" in self.args.env_name:
                    obs_video += 0.5
                wandb_lib.log_video(f"videos/obs/{k}", obs_video, i)
            else:
                wandb_lib.log_histogram(f"training/observations/{k}", observations[k].float(), i)

        return i

    def train_step(self, batch_data, i):
        self.model.train()
        self.optim.zero_grad()

        # Unpackd data
        advantages = batch_data["advantages"]
        selected_prob = batch_data["policy_probs"]
        values = batch_data["values"]
        rewards_to_go = batch_data["rewards_to_go"]
        obs = {k.removeprefix("obs__"): v for k, v in batch_data.items() if k.startswith("obs__")}
        actions = {k.removeprefix("action__"): v for k, v in batch_data.items() if k.startswith("action__")}

        # Normalize the advantages.
        # Note the advantages are normalized by batch, not the entire epoch altogether
        # might want to parameterize the constant here. I had it at 1e-5 originally
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        values_new, dist_new = self.model(obs)
        selected_prob_new = dist_new.log_prob(actions)

        # Compute the PPO loss
        prob_ratio = torch.exp(selected_prob_new - selected_prob)
        a = prob_ratio * advantages
        b = torch.clamp(prob_ratio, 1 - self.args.clip_range, 1 + self.args.clip_range) * advantages
        ppo_loss = -1 * torch.min(a, b)
        ppo_loss = ppo_loss.mean()

        # Compute the value function loss
        # Clipped loss - same idea as PPO loss, don't allow value to move too
        # far from where it was previously
        if self.args.clip_range_vf is None:
            value_pred_clipped = values_new
        else:
            value_pred_clipped = values + (values_new - values).clamp(
                -self.args.clip_range_vf, self.args.clip_range_vf
            )

        # NOTE: implementations differ in how they handle this.
        # this is how I had it previously (presumably pulled from original baselines??)
        # value_losses = (values_new - rewards_to_go) ** 2
        # value_losses_clipped = (value_pred_clipped - rewards_to_go) ** 2
        # value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
        # value_loss = value_loss.mean()
        # but sb3 uses a simpler method:
        value_loss = F.mse_loss(rewards_to_go, value_pred_clipped)

        entropy_loss = torch.mean(dist_new.entropy())

        loss = ppo_loss + self.args.value_loss_coef * value_loss - self.args.entropy_coef * entropy_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
        self.optim.step()

        # Lots of logging
        clip_fraction = torch.mean((torch.abs(prob_ratio - 1) > self.args.clip_range).float()).item()
        wandb_lib.log_scalar("loss/clip_fraction", clip_fraction, i)
        wandb_lib.log_histogram("loss/ppo_loss", -1 * torch.min(a, b), i)
        wandb_lib.log_scalar("loss/value_loss", value_loss, i)
        wandb_lib.log_scalar("loss/entropy_loss", -self.args.entropy_coef * entropy_loss, i)
        wandb_lib.log_scalar("loss/loss", loss, i)
        wandb_lib.log_histogram("training/value_pred", values_new, i)
        wandb_lib.log_histogram("training/advantage", advantages, i)
        wandb_lib.log_histogram("training/value_target", rewards_to_go, i)
        wandb_lib.log_histogram("training/prob_ratio", prob_ratio, i)
        wandb_lib.log_scalar("policy/policy entropy", entropy_loss, i)

        # Log actions
        for k, space in self.action_space.spaces.items():
            if isinstance(space, OneHotMultiDiscrete):
                for act_i in range(len(space.nvec)):
                    for cat_i in range(space.nvec[act_i]):
                        wandb_lib.log_histogram(f"actions/{k}_{act_i}_{cat_i}", actions[k][:, act_i, cat_i].float(), i)
                        if space.nvec[act_i] == 2:
                            # a binary space only needs one of the categories logged
                            break
            elif isinstance(space, gym.spaces.Box):
                assert len(space.shape) == 1
                for act_i in range(space.shape[0]):
                    wandb_lib.log_histogram(f"actions/{k}_{act_i}", actions[k][:, act_i].float(), i)
