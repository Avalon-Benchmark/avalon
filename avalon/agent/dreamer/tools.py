from typing import Any
from typing import Callable
from typing import Dict
from typing import List

import numpy as np
import torch
import tree
from torch import Tensor
from torch.nn import functional as F
from tree import map_structure

# "TanhBijector" and "SampleDist" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch


class SampleDist(torch.distributions.Distribution):
    def __init__(self, dist: torch.distributions.Distribution, samples: int = 100) -> None:
        self._dist = dist
        self._samples = samples

    @property
    def name(self) -> str:
        return "SampleDist"

    @property
    def mean(self) -> Tensor:
        sample = self._dist.rsample()
        return torch.mean(sample, 0)

    def mode(self) -> Tensor:
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self) -> Tensor:
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def log_prob(self, sample: Tensor) -> Tensor:
        return self._dist.log_prob(sample)  # type: ignore

    def sample(self) -> Tensor:  # type: ignore
        return self._dist.sample().detach()  # type: ignore

    def rsample(self) -> Tensor:  # type: ignore
        return self._dist.rsample()  # type: ignore


class TanhBijector(torch.distributions.Transform):
    def __init__(self) -> None:
        super().__init__()
        self.bijective = True
        self.domain = torch.distributions.constraints.real
        self.codomain = torch.distributions.constraints.interval(-1.0, 1.0)

    @property
    def sign(self) -> float:
        return 1.0

    def _call(self, x: Tensor) -> Tensor:
        return torch.tanh(x)

    def _inverse(self, y: Tensor) -> Tensor:
        # TODO: cast this to fp32 if we do half-precision
        y = torch.where((torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y)
        y = torch.atanh(y)
        return y

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return 2.0 * (np.log(2) - x - F.softplus(-2.0 * x))  # type: ignore


def lambda_return(
    reward: Tensor, value: Tensor, pcont: Tensor, bootstrap: Tensor, lambda_: float, axis: int
) -> Tensor:
    """This is dreamer's value backup method. Similar to GAE - is it the same?
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    """
    assert axis == 0
    assert isinstance(pcont, torch.Tensor)
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    assert bootstrap is not None
    # tensor[None] just adds a dim of len 1 at the beginning of the tensor. so x, y, z => 1, x, y z
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    # Here the map structures are: tuple of tensors, tensor.
    returns: Tensor = static_scan(
        lambda agg, cur: cur[0] + cur[1] * lambda_ * agg, (inputs, pcont), bootstrap, reverse=True
    )
    return returns


def static_scan(fn: Callable, inputs: Any, start: Any, reverse: bool = False) -> Any:
    """This is a utility from danijar for applying a function iteratively over each element in a sequence,
    combined with the current value of an accumulator. It's quite confusing but also quite handy."""
    last = start
    outputs: List[List] = [[] for _ in tree.flatten(start)]
    indices = list(range(len(tree.flatten(inputs)[0])))
    if reverse:
        indices = list(reversed(indices))
    for index in indices:
        inp = map_structure(lambda x: x[index], inputs)
        last = fn(last, inp)
        for o, l in zip(outputs, tree.flatten(last)):
            o.append(l)
    if reverse:
        outputs = [list(reversed(x)) for x in outputs]
    outputs_stacked = [torch.stack(x, 0) for x in outputs]
    return tree.unflatten_as(start, outputs_stacked)


def pack_list_of_dicts(data: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    out = {}
    for key in data[0].keys():
        out[key] = torch.stack([x[key] for x in data], dim=0)
    return out
