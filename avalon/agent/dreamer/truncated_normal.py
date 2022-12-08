"""this implementation is from https://github.com/toshas/torch_truncnorm/blob/main/TruncatedNormal.py"""
import math
from numbers import Number
from typing import Optional
from typing import Union

import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)

TensorOrNumber = Union[Tensor, float, int]


class TruncatedStandardNormal(Distribution):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        "a": constraints.real,
        "b": constraints.real,
    }
    has_rsample = True

    def __init__(self, a: TensorOrNumber, b: TensorOrNumber, validate_args: Optional[bool] = None) -> None:
        self.a: Tensor
        self.b: Tensor
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError("Truncation bounds types are different")
        if any((self.a >= self.b).view(-1).tolist()):
            raise ValueError("Incorrect truncation range")
        eps = torch.finfo(self.a.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        # This explodes (becomes 1, which explodes stuff downstream) if a > ~4
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z: Tensor = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (
            self._little_phi_b * little_phi_coeff_b - self._little_phi_a * little_phi_coeff_a
        ) / self._Z
        self._mean: Tensor = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance: Tensor = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        self._entropy: Tensor = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z
        self._mode: Tensor = torch.clip(torch.zeros_like(self.a), self.a, self.b)

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self) -> Tensor:
        return self._mean

    def mode(self) -> Tensor:
        return self._mode

    @property
    def variance(self) -> Tensor:
        return self._variance

    @property
    def stddev(self):
        return self.variance.pow(0.5)

    # @property
    def entropy(self) -> Tensor:
        return self._entropy

    @property
    def auc(self) -> Tensor:
        return self._Z

    @staticmethod
    def _little_phi(x: Tensor) -> Tensor:
        return (-(x**2) * 0.5).exp() * CONST_INV_SQRT_2PI  # type:  ignore

    @staticmethod
    def _big_phi(x: Tensor) -> Tensor:
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x: Tensor) -> Tensor:
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value: Tensor) -> Tensor:
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value**2) * 0.5  # type: ignore

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        sample = self.icdf(p)
        assert torch.all(torch.isfinite(sample))
        return sample


class TruncatedNormal(TruncatedStandardNormal):
    """
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    has_rsample = True

    def __init__(
        self,
        loc: TensorOrNumber,
        scale: TensorOrNumber,
        low: TensorOrNumber,
        high: TensorOrNumber,
        validate_args: Optional[bool] = None,
    ) -> None:
        self.loc: Tensor
        self.scale: Tensor
        self.low: Tensor
        self.high: Tensor
        self.loc, self.scale, self.low, self.high = broadcast_all(loc, scale, low, high)
        a = (self.low - self.loc) / self.scale
        b = (self.high - self.loc) / self.scale
        assert torch.all((b - a) > 0)
        super(TruncatedNormal, self).__init__(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale**2
        self._entropy += self._log_scale
        self._mode = torch.clip(self.loc, self.low, self.high)

    def _to_std_rv(self, value: Tensor) -> Tensor:
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value: Tensor) -> Tensor:
        return value * self.scale + self.loc

    def cdf(self, value: Tensor) -> Tensor:
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value: Tensor) -> Tensor:
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value: Tensor) -> Tensor:
        return super(TruncatedNormal, self).log_prob(self._to_std_rv(value)) - self._log_scale


class ClippedTruncatedNormal(TruncatedNormal):
    """Matches the Dreamerv2 impl, although I don't get what this is for/why it's necessary."""

    def __init__(
        self, loc: TensorOrNumber, scale: TensorOrNumber, low: TensorOrNumber, high: TensorOrNumber, clip: float = 1e-6
    ) -> None:
        super().__init__(loc, scale, low, high)
        self.lower_clip: Tensor = low + clip  # type: ignore
        self.upper_clip: Tensor = high - clip  # type: ignore

    def rsample(self, *args, **kwargs) -> Tensor:  # type: ignore
        event = super().rsample(*args, **kwargs)
        # TODO: what's this clipping for? applied after drawing samples, just slight pulls in samples
        # that are too close to the bounds. only applied in forward pass. but why?
        # would make sense if the actions are being arctanh'ed for some reason and the edges have really steep slope.
        # but this doesn't seem common/logical?
        # or maybe this is just to ensure we don't pick a value exactly equal to the limit (is that even possible?)
        clipped = torch.clip(event, self.lower_clip, self.upper_clip)
        # Apply clipping in forward but not backward pass.
        event = event - event.detach() + clipped.detach()
        return event

    def sample(self, *args, **kwargs) -> Tensor:  # type: ignore
        return self.rsample(*args, **kwargs).detach()
