from __future__ import annotations

from typing import Dict
from typing import Optional

import gym
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.distributions import Independent
from torch.distributions import Normal
from torch.distributions import OneHotCategorical
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.nn import functional as F

from avalon.agent.common import wandb_lib
from avalon.agent.common.params import ClippedNormalMode
from avalon.agent.common.params import Params
from avalon.agent.common.types import ActionBatch
from avalon.agent.common.wrappers import OneHotMultiDiscrete
from avalon.agent.dreamer.tools import SampleDist
from avalon.agent.dreamer.tools import TanhBijector
from avalon.agent.dreamer.truncated_normal import ClippedTruncatedNormal


class NormalWithMode(Normal):
    def mode(self) -> Tensor:
        mean = self.mean
        assert isinstance(mean, Tensor)
        return mean

    def __getitem__(self, item):
        # slicing
        assert isinstance(item, (int, tuple))
        return NormalWithMode(self.loc[item], self.scale[item])


class IndependentWithMode(Independent):
    def mode(self) -> Tensor:
        mode = self.base_dist.mode()
        assert isinstance(mode, Tensor)
        return mode

    def __getitem__(self, item):
        # slicing
        assert isinstance(item, (int, tuple))
        if isinstance(item, tuple):
            assert len(item) == 1  # this could be lifted
        assert self.reinterpreted_batch_ndims == 1
        # if this was an Independent(1), then slicing just removes this wrapper?
        return self.base_dist[item]


class NormalHead(nn.Module):
    """A module that builds a Diagonal Gaussian distribution from means.

    If model_provides_std=False, standard deviations are learned parameters in this module.
    Otherwise they are taken as inputs.
    """

    def __init__(
        self,
        action_space: gym.spaces.Box,
        init_std: float,
        min_std: float,
        clipped_normal_mode: ClippedNormalMode,
        model_provides_std: bool = True,
    ) -> None:
        super().__init__()
        assert isinstance(action_space, gym.spaces.Box)
        assert len(action_space.shape) == 1
        assert np.all(action_space.low == -1.0)
        assert np.all(action_space.high == 1.0)
        self.num_outputs = action_space.shape[0]
        self.model_provides_std = model_provides_std
        self.min_std = min_std
        self.clipped_normal_mode = clipped_normal_mode
        if model_provides_std:
            # We take 2 values per output element, one for mean and one for (raw) std
            self.num_inputs = self.num_outputs * 2
            self.init_std = init_std
        else:
            # # We'll use a constant learned std
            # self.num_inputs = self.num_outputs
            # # initial variance is e^0 = 1
            # self.log_std = nn.Parameter(torch.zeros(self.num_outputs))
            raise NotImplementedError

    def forward(self, x: Tensor) -> torch.distributions.Distribution:
        # x should have shape (..., action_dim)
        assert x.shape[-1] == self.num_inputs
        if self.model_provides_std:
            mean, raw_std = torch.chunk(x, 2, -1)
        else:
            raise NotImplementedError
            # mean = x
            # std = self.log_std.exp() + self.min_std

        if self.clipped_normal_mode == ClippedNormalMode.NO_CLIPPING:
            # note that standard practice would be to use std = log_std.exp(), here we use a centered softplus instead
            raw_init_std = np.log(np.exp(self.init_std) - 1)  # such that init_std = softplus(raw_init_std)
            std = F.softplus(raw_std + raw_init_std) + self.min_std
            dist: torch.distributions.Distribution = NormalWithMode(mean, std)
            return IndependentWithMode(dist, reinterpreted_batch_ndims=1)
        elif self.clipped_normal_mode == ClippedNormalMode.SAMPLE_DIST:
            # This is exactly matching the dreamerv1 (need to double check) and v2-"tanh-normal" mode
            # this mean_scale and tanh thing is a sort of soft clipping of the input.
            mean_scale = 5
            mean_scaled = mean_scale * torch.tanh(mean / mean_scale)
            # note that standard practice would be to use std = log_std.exp(), here we use a centered softplus instead
            raw_init_std = np.log(np.exp(self.init_std) - 1)  # such that init_std = softplus(raw_init_std)
            std = F.softplus(raw_std + raw_init_std) + self.min_std
            dist = Normal(mean_scaled, std)
            transformed_dist = TransformedDistribution(dist, TanhBijector())
            independent_dist = Independent(transformed_dist, 1)
            sample_dist = SampleDist(independent_dist)
            return sample_dist
        elif self.clipped_normal_mode == ClippedNormalMode.TRUNCATED_NORMAL:
            # This is exactly the default mode for dreamerv2.
            # These are the dreamer default params
            # assert self.init_std == 0
            # assert self.min_std == 0.1
            # To be precise, we should inverse-sigmoid init_std, but this is how dreamerv2 did it.
            std = 2 * torch.sigmoid((raw_std + self.init_std) / 2) + self.min_std
            dist = ClippedTruncatedNormal(torch.tanh(mean), std, -1, 1)
            dist = IndependentWithMode(dist, 1)
            return dist
        assert False


class StraightThroughOneHotCategorical(OneHotCategorical):
    def rsample(self, sample_shape: torch.Size = torch.Size()):
        assert sample_shape == torch.Size()
        # Straight through biased gradient estimator.
        sample = self.sample(sample_shape).to(torch.float32)
        probs = self.probs
        assert sample.shape == probs.shape
        sample += probs - probs.detach()
        return sample.float()

    def mode(self) -> Tensor:
        return F.one_hot(self.probs.argmax(dim=-1), self.event_shape[-1]).float()  # type: ignore

    def __getitem__(self, item):
        # slicing
        assert isinstance(item, (int, tuple))
        return StraightThroughOneHotCategorical(self.logits[item])


class MultiCategoricalHead(nn.Module):
    """Represents multiple categorical dists. All must have the same number of categories."""

    def __init__(self, num_actions: int, num_categories: int, center_and_clamp_logits: bool = False) -> None:
        super().__init__()
        self.num_categories = num_categories
        self.num_actions = num_actions
        self.num_inputs = num_categories * num_actions
        self.num_outputs = num_categories * num_actions
        self.center_and_clamp_logits = center_and_clamp_logits

    def forward(self, x: Tensor) -> torch.distributions.Distribution:
        assert x.shape[-1] == self.num_actions * self.num_categories
        x = rearrange(x, "... (a c) -> ... a c", a=self.num_actions, c=self.num_categories)
        if self.center_and_clamp_logits:
            # This was found to consistently hurt performance across PPO Procgen envs,
            # but was used (and presumably helped at some point) in Avalon
            x = x - x.mean(dim=-1, keepdim=True)
            x = torch.clamp(x, -4, 4)
        dist = StraightThroughOneHotCategorical(logits=x)
        # TODO: i'm not sure this is right if there's multiple actions in this MultiCategorical dist.
        independent_dist = IndependentWithMode(dist, 1)
        return independent_dist


class DictActionHead(torch.nn.Module):
    """This model handles generating a policy distribution from latents.

    Latents should be passed in with shape (..., self.num_inputs), and a DictActionDist will be returned.
    """

    def __init__(self, action_space: gym.spaces.Dict, params: Params) -> None:
        super().__init__()
        self.action_space = action_space

        # Build action heads
        self.num_inputs = 0
        self.num_outputs = 0
        action_heads = {}
        for k, space in self.action_space.spaces.items():
            if isinstance(space, gym.spaces.Box):
                # The input is (..., num_actions)
                # The output of sample() should be (..., num_actions)
                head: torch.nn.Module = NormalHead(
                    space,
                    init_std=params.policy_normal_init_std,
                    min_std=params.policy_normal_min_std,
                    clipped_normal_mode=params.clipped_normal_mode,
                    model_provides_std=params.normal_std_from_model,
                )
            elif isinstance(space, OneHotMultiDiscrete):
                # A MultiDiscrete is multiple independent Discrete spaces.
                # We coerce all discrete space types into this with a wrapper
                # This won't work if the discretes have different num_categories.
                assert len(set(space.nvec)) == 1
                head = MultiCategoricalHead(
                    num_actions=len(space.nvec),
                    num_categories=space.max_categories,
                    center_and_clamp_logits=params.center_and_clamp_discrete_logits,
                )
            else:
                assert False
            action_heads[k] = head
            self.num_inputs += action_heads[k].num_inputs  # type: ignore
            self.num_outputs += action_heads[k].num_outputs  # type: ignore
        self.action_heads = torch.nn.ModuleDict(action_heads)

    def forward(self, action_logits: Tensor) -> "DictActionDist":
        dists = {}
        i = 0
        for k, head in self.action_heads.items():
            logits = action_logits[..., i : i + head.num_inputs]
            dists[k] = head(logits)
            i += head.num_inputs
        return DictActionDist(dists)


class DictActionDist(torch.distributions.Distribution):
    """This is an instance of a torch Distribution that holds key-value pairs of other Distributions.

    It's used for e.g. sampling from an entire Dict action space in one operation,
    which will return a dictionary of samples.

    Operations like entropy() will reduce over all dists to return a single value (per batch element)."""

    def __init__(self, dists: dict[str, torch.distributions.Distribution]) -> None:
        super().__init__(validate_args=False)
        self.dists = dists

    def sample(self) -> ActionBatch:  # type: ignore
        # Detach is just for good measure in case someone messes up the sample() impl :)
        return {k: v.sample().detach() for k, v in self.dists.items()}

    def rsample(self) -> ActionBatch:  # type: ignore
        return {k: v.rsample() for k, v in self.dists.items()}

    def log_prob(self, actions: dict[str, Tensor]) -> Tensor:
        """Compute the log prob of the given action under the given dist (batchwise).

        Log prob is a scalar (per batch element.)"""
        # actions is a dict of tensors of shape (batch_size, num_outputs)
        log_probs = []
        for k, dist in self.dists.items():
            # batch_size = actions[k].shape[0]
            # assert actions[k].shape == (batch_size, self.action_heads[k].num_outputs)
            log_prob = dist.log_prob(actions[k])
            # The log_prob is over the entire action space, so it reduces to a single scalar per batch element
            # assert log_prob.shape == (batch_size,)
            log_probs.append(log_prob)

        # the output should have shape (batch_size, )
        return torch.stack(log_probs, dim=1).sum(dim=1)

    def entropy(self) -> Tensor:
        # The entropy is over the entire action space, so it reduces to a single scalar per batch element
        entropies = [v.entropy() for v in self.dists.values()]
        return torch.stack(entropies, dim=1).sum(dim=1)

    def mean(self) -> Dict[str, Tensor]:
        return {k: v.mean() for k, v in self.dists.items()}

    def mode(self) -> Dict[str, Tensor]:
        return {k: v.mode() for k, v in self.dists.items()}

    def __getitem__(self, item: str):
        if isinstance(item, str):
            # user is treating this as a dict
            return self.dists[item]
        elif isinstance(item, (int, tuple)):
            # user is slicing in the batch dim
            return DictActionDist({k: v[item] for k, v in self.dists.items()})
        else:
            raise ValueError

    def __iter__(self):
        # in case anyone tries to iterate on this (cough cough wandb.watch)
        return self.dists.values()


def visualize_action_dists(
    action_space: gym.spaces.Dict,
    action_dist_dict: DictActionDist,
    prefix: str = "action_dists",
    freq: Optional[int] = None,
):
    """Log the action dists to wandb. Any batch shape is fine.

    The action dists often give more info than the actual actions, since the actions are sampled from these dists.
    """
    for k, space in action_space.spaces.items():
        action_dist = action_dist_dict[k]
        if isinstance(space, OneHotMultiDiscrete):
            # We coerce all discrete spaces into this type now
            # action_probs should have shape (batch_dims, num_actions, num_categories)
            # The dist here is an IndependentWithMode(StraightThroughOneHotCategorical())
            probs = action_dist.base_dist.probs
            probs = probs.reshape(-1, probs.shape[-2], probs.shape[-1])
            for act_i in range(len(space.nvec)):
                for cat_i in range(int(space.nvec[act_i])):
                    wandb_lib.log_histogram(
                        f"{prefix}/{k}_{act_i}_{cat_i}", probs[:, act_i, cat_i], mean_freq=freq, hist_freq=freq
                    )
                    if space.nvec[act_i] == 2:
                        # a binary space only needs one of the categories logged
                        break
        elif isinstance(space, gym.spaces.Box):
            assert len(space.shape) == 1
            if isinstance(action_dist, SampleDist):
                raise NotImplementedError
            elif isinstance(action_dist, IndependentWithMode):
                action_dist = action_dist.base_dist
                assert isinstance(action_dist, (ClippedTruncatedNormal, NormalWithMode))
                means = action_dist.mean
                for dim in range(means.shape[-1]):
                    wandb_lib.log_histogram(
                        f"{prefix}/{k}_{dim}_mean", means[..., dim], mean_freq=freq, hist_freq=freq
                    )
                stds = action_dist.stddev
                for dim in range(means.shape[-1]):
                    wandb_lib.log_histogram(f"{prefix}/{k}_{dim}_std", stds[..., dim], mean_freq=freq, hist_freq=freq)


def visualize_actions(
    action_space: gym.spaces.Dict, action_dict: dict[str, Tensor], prefix: str = "actions", freq: Optional[int] = None
) -> None:
    """Log a batch of actions to wandb (any batch shape is fine)."""
    for k, space in action_space.spaces.items():
        action = action_dict[k]
        if isinstance(space, OneHotMultiDiscrete):
            # expect one-hot of shape (batch_dims, num_actions, num_categories)
            # flatten the batch dim
            action = action.reshape(-1, action.shape[-2], action.shape[-1])
            for act_i in range(len(space.nvec)):
                for cat_i in range(int(space.nvec[act_i])):
                    # Histograms don't make sense, we'll just log the % of actions where this category is selected
                    wandb_lib.log_scalar(
                        f"{prefix}/{k}_{act_i}_{cat_i}", action[:, act_i, cat_i].float().mean(), freq=freq
                    )
                    if space.nvec[act_i] == 2:
                        # a binary space only needs one of the categories logged
                        break
        elif isinstance(space, gym.spaces.Box):
            assert len(space.shape) == 1
            # expect shape (batch_dims, num_actions)
            for act_i in range(space.shape[0]):
                wandb_lib.log_histogram(
                    f"{prefix}/{k}_{act_i}", action[:, act_i].float(), mean_freq=freq, hist_freq=freq
                )
