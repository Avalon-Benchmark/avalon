from typing import Dict

import gym
import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from torch import nn as nn
from torch.distributions import Independent
from torch.distributions import Normal
from torch.distributions import OneHotCategorical
from torch.nn import functional as F

from agent.wrappers import OneHotMultiDiscrete


class NormalWithMode(Normal):
    def mode(self):
        return self.mean


class NormalHead(nn.Module):
    """A module that builds a Diagonal Gaussian distribution from means.

    If model_provides_std=False, standard deviations are learned parameters in this module.
    Otherwise they are taken as inputs.
    """

    def __init__(
        self, action_space: gym.spaces.Box, model_provides_std: bool = False, min_std: float = 0.01, mode: str = "none"
    ):
        super().__init__()
        assert isinstance(action_space, gym.spaces.Box)
        assert len(action_space.shape) == 1
        assert np.all(action_space.low == -1.0)
        assert np.all(action_space.high == 1.0)
        self.num_outputs = action_space.shape[0]
        self.min_std = min_std
        self.num_inputs = self.num_outputs * 2
        self.init_std = 1
        self.raw_init_std = np.log(np.exp(self.init_std) - 1)  # such that init_std = softplus(raw_init_std)

    def forward(self, x: Tensor) -> torch.distributions.Distribution:
        # x should have shape (..., action_dim)
        assert x.shape[-1] == self.num_inputs
        mean, raw_std = torch.chunk(x, 2, -1)
        # note that standard practice would be to use std = log_std.exp(), here we use a centered softplus instead
        std = F.softplus(raw_std + self.raw_init_std) + self.min_std
        dist = NormalWithMode(mean, std)
        # TODO: really should make a DiagonalNormal class vs screwing with these Independents all the time.
        return IndependentWithMode(dist, reinterpreted_batch_ndims=1)


class StraightThroughOneHotCategorical(OneHotCategorical):
    def rsample(self, sample_shape: torch.Size = torch.Size()):
        # TODO: allow other sample_shapes
        assert sample_shape == torch.Size()
        # Straight through biased gradient estimator.
        sample = self.sample(sample_shape).to(torch.float32)
        probs = self.probs
        assert sample.shape == probs.shape
        sample += probs - probs.detach()
        return sample.float()

    def mode(self) -> Tensor:
        return F.one_hot(self.probs.argmax(dim=-1), self.event_shape[-1]).float()


class IndependentWithMode(Independent):
    def mode(self):
        return self.base_dist.mode()


class MultiCategoricalHead(nn.Module):
    """Represents multiple categorical dists. All must have the same number of categories."""

    def __init__(self, num_actions: int, num_categories: int, is_scalar: bool = False):
        """is_scalar=True turns this into a normal Categorical with no additional axis for num_actions."""
        # TODO: should be able to clean out the is_scalar logic now
        super().__init__()
        self.num_categories = num_categories
        self.is_scalar = is_scalar
        if is_scalar:
            assert self.num_actions == 1
        self.num_actions = num_actions
        self.num_inputs = num_categories * num_actions
        self.num_outputs = num_categories * num_actions

    def forward(self, x: Tensor) -> torch.distributions.Distribution:
        assert x.shape[-1] == self.num_actions * self.num_categories
        if not self.is_scalar:
            x = rearrange(x, "... (a c) -> ... a c", a=self.num_actions, c=self.num_categories)
        x = x - x.mean(dim=-1, keepdim=True)
        x = torch.clamp(x, -4, 4)
        dist = StraightThroughOneHotCategorical(logits=x)
        if not self.is_scalar:
            dist = Independent(dist, 1)
        return dist


class DictActionHead(torch.nn.Module):
    """This model handles generating a policy distribution from latents.

    Latents should be passed in with shape (..., self.num_inputs), and a DictActionDist will be returned.
    """

    def __init__(self, action_space: gym.spaces.Dict, args):
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
                head = NormalHead(space, model_provides_std=args.normal_std_from_model, mode=args.clipped_normal_mode)
            elif isinstance(space, OneHotMultiDiscrete):
                assert len(set(space.nvec)) == 1
                head = MultiCategoricalHead(num_actions=len(space.nvec), num_categories=space.max_categories)
            else:
                assert False
            action_heads[k] = head
            self.num_inputs += action_heads[k].num_inputs
            # TODO: i don't think num_outputs is used?
            self.num_outputs += action_heads[k].num_outputs
        self.action_heads = torch.nn.ModuleDict(action_heads)

    def forward(self, action_logits: Tensor) -> "DictActionDist":
        # batch_size = action_logits.shape[0]
        # assert action_logits.shape == (batch_size, self.num_inputs)
        # can also be (b, t, num_inputs)
        dists = {}
        # NOTE: this depends on the fact that dicts are ordered in recent CPython.
        i = 0
        for k, head in self.action_heads.items():
            # we want the action logit to have shape (..., num_inputs)
            logits = action_logits[..., i : i + head.num_inputs]
            dists[k] = head(logits)
            i += head.num_inputs
        return DictActionDist(dists)


class DictActionDist(torch.distributions.Distribution):
    """This is an instance of a torch Distribution that holds key-value pairs of other Distributions.

    It's used for e.g. sampling from an entire Dict action space in one operation,
    which will return a dictionary of samples.

    Operations like entropy() will reduce over all dists to return a single value (per batch element)."""

    def __init__(self, dists: Dict[str, torch.distributions.Distribution]):
        super().__init__(validate_args=False)
        self.dists = dists
        # self.action_heads = action_heads

    def sample(self) -> Dict[str, Tensor]:
        """Returns an action as a dict of batched tensors."""
        # rsample is not used - gradients propagate thru log_prob, not sample
        out = {}
        for k, v in self.dists.items():
            # should have shape (batch_size, num_outputs)
            # Detach is just for good measure in case someone messes up the sample() impl :)
            sample = v.sample().detach()
            # assert len(sample.shape) == 2
            assert sample.dtype == torch.float32
            # assert sample.shape[1] == self.action_heads[k].num_outputs
            out[k] = sample
        return out

    def log_prob(self, actions: Dict[str, Tensor]) -> Tensor:
        """Compute the log prob of the given action under the given dist (batchwise).

        Log prob is a scalar (per batch element.)"""
        # actions is a dict of tensors of shape (batch_size, num_outputs)
        log_probs = []
        for k, dist in self.dists.items():
            # batch_size = actions[k].shape[0]
            # assert actions[k].shape == (batch_size, self.action_heads[k].num_outputs)
            # TODO: review why the float() call is needed here (for the Bernoulli dist)
            # how do we get a non-float here??
            log_prob = dist.log_prob(actions[k].float())
            # The log_prob is over the entire action space, so it reduces to a single scalar per batch element
            # assert log_prob.shape == (batch_size,)
            log_probs.append(log_prob)

        # log probs has shape [num_actions, batch_size].
        return torch.stack(log_probs, dim=1).sum(dim=1)

    def entropy(self) -> Tensor:
        entropies = []
        for k, dist in self.dists.items():
            entropy = dist.entropy()
            # # The entropy is over the entire action space, so it reduces to a single scalar per batch element
            # assert len(entropy.shape) == 1
            entropies.append(entropy)
        return torch.stack(entropies, dim=1).sum(dim=1)

    def mean(self):
        return {k: v.mean() for k, v in self.dists.items()}

    def mode(self):
        return {k: v.mode() for k, v in self.dists.items()}

    def rsample(self):
        return {k: v.rsample() for k, v in self.dists.items()}
