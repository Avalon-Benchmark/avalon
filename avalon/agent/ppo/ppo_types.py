from __future__ import annotations

import attr
from torch import Tensor

from avalon.agent.common.types import BatchData
from avalon.agent.common.types import BatchSequenceData
from avalon.agent.common.types import SequenceData
from avalon.agent.common.types import StepData


@attr.s(auto_attribs=True, frozen=True)
class PPOBatchSequenceData(BatchSequenceData):
    value: Tensor
    policy_prob: Tensor
    policy_entropy: Tensor


@attr.s(auto_attribs=True, frozen=True)
class PPOSequenceData(SequenceData):
    value: Tensor
    policy_prob: Tensor
    policy_entropy: Tensor


@attr.s(auto_attribs=True, frozen=True)
class PPOBatchSequenceDataWithGAE(PPOBatchSequenceData):
    # These get added after GAE computation
    advantage: Tensor
    reward_to_go: Tensor


PPOBatchDataWithGAE = PPOBatchSequenceDataWithGAE  # these have shape (batch, ...) instead of (batch, timesteps, ...)


@attr.s(auto_attribs=True, frozen=True)
class PPOBatchData(BatchData):
    value: Tensor
    policy_prob: Tensor
    policy_entropy: Tensor


@attr.s(auto_attribs=True, frozen=True)
class PPOStepData(StepData):
    batch_type = PPOBatchData
    sequence_type = PPOSequenceData  # type: ignore
    batch_sequence_type = PPOBatchSequenceData  # type: ignore
    value: float
    policy_prob: float
    policy_entropy: float
