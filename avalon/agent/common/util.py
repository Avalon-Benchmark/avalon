import os
import pickle
import warnings
from functools import partial
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import Union

import attr
import gym
import numpy as np
import torch
from gym.spaces import Box
from numpy.typing import NDArray
from torch import Tensor
from tree import map_structure

from avalon.agent.common import wandb_lib
from avalon.agent.common.wrappers import OneHotMultiDiscrete
from avalon.common.type_utils import assert_not_none
from avalon.contrib.utils import make_deterministic

numpy_to_torch_dtype = {
    bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
}


def get_checkpoint_file(protocol_path: str) -> str:
    """Gets a string path to the checkpoint file.
    - protocol_path:
        a string specifying the possible places to resume from, either local file or wandb run
        valid formats include:
            wandb://{project}/{run}/{file_name}
            file://{absolute_file_path}
        for example:
            wandb://untitled-ai/sf3189ytcfg/checkpoint.pt
            file:///home/user/runs/good_run/checkpoint.pt
    """
    protocol, path = protocol_path.split("://")
    if protocol == "wandb":
        project, run_id, filename = path.split("/")
        return wandb_lib.download_file(run_id, project, filename)
    elif protocol == "file":
        return path
    else:
        raise ValueError(f"protocol {protocol} in {protocol_path} is not currently supported")


PackedType = TypeVar("PackedType")


def pack_1d_list(sequence: List, out_cls: Type[PackedType]) -> PackedType:
    """Pack a list of StepDatas into a BatchData, or a list of SequenceDatas into a BatchSequenceData"""
    out: dict[str, Any] = {}
    for field_obj in attr.fields(type(sequence[0])):
        field = field_obj.name
        if field == "info":
            continue
        sample = getattr(sequence[0], field)
        if isinstance(sample, dict):
            out[field] = {
                k: torch.stack([getattr(transition, field)[k] for transition in sequence]) for k in sample.keys()
            }
        elif np.isscalar(sample):
            out[field] = torch.tensor(np.stack([getattr(transition, field) for transition in sequence]))
        else:
            # tensor
            out[field] = torch.stack([getattr(transition, field) for transition in sequence])
    return out_cls(**out)


def pack_2d_list(batch: List[List], out_cls: Type[PackedType]) -> PackedType:
    """Pack a batch of StepDatas into a BatchSequenceData (or subclass)"""
    out: dict[str, Any] = {}
    for k_obj in attr.fields(type(batch[0][0])):
        k = k_obj.name
        if k == "info":
            continue
        example = batch[0][0].__getattribute__(k)
        if np.isscalar(example):
            out[k] = torch.tensor(
                np.stack([np.stack([getattr(transition, k) for transition in trajectory]) for trajectory in batch])
            )
        elif isinstance(example, Tensor):
            out[k] = torch.stack(
                [torch.stack([getattr(transition, k) for transition in trajectory]) for trajectory in batch]
            )
        elif isinstance(example, dict):
            out2: dict[str, Any] = {}
            for k2 in example.keys():
                out2[k2] = torch.stack(
                    [torch.stack([getattr(transition, k)[k2] for transition in trajectory]) for trajectory in batch]
                )
            out[k] = out2
        else:
            assert False
    return out_cls(**out)


def postprocess_uint8_to_float(
    data: Dict[str, torch.Tensor], center: bool = True, observation_prefix: str = ""
) -> Dict[str, torch.Tensor]:
    """Convert uint8 (0,255) to float (-.5, .5) in a dictionary of rollout data.

    We use this to keep images as uint8 in storage + transfer.
    If observation_prefix is passed, then only keys with that prefix will be looked at for transforming.
    This allows handling a whole batch of rollout data (with actions, values, etc), which is convenient.
    """
    out = {}
    for k, v in data.items():
        if k.startswith(observation_prefix) and v.dtype in (np.uint8, torch.uint8):
            v = v / 255.0
            if center:
                v = v - 0.5
        out[k] = v
    return out


ArrayType = Union[NDArray, torch.Tensor]


def hash_tensor(x: Tensor, data_only: bool = False) -> str:
    """Get a unique hash for a tensor.
    If `include_metadata` is False, the hash is based only on the data itself (and dtype, and shape).
    If `include_metadata` is True, the hash includes all things known to make 2 tensors act differently.
    """
    to_hash = str(pickle.dumps(torch.clone(x.detach().cpu(), memory_format=torch.contiguous_format).numpy()))
    if not data_only:
        to_hash += str(x.stride())  # stride matters for determinism and isn't otherwise checked
        to_hash += str(x.device)
        to_hash += str(x.dtype)
    return str(hash(to_hash))[:8]


def debug_hash_tensor(name: str, x: Tensor, concise: bool = True, print_tensor: bool = False) -> None:
    """Print useful output to help understand why two tensors don't have the same hash."""
    torch.set_printoptions(precision=8)
    data_only = torch.clone(x.cpu(), memory_format=torch.contiguous_format)
    print(
        f"{name}; full hash: {hash_tensor(x)}; data hash: {hash_tensor(data_only, data_only=True)}; shape: {x.shape}; dtype: {x.dtype}; device: {x.device}"
    )
    if not concise:
        print("sum", x.sum())
        print("device", x.device)
        print("stride", x.stride())
        print("max", x.max())
        print("min", x.min())
    if print_tensor:
        print(x)


class HashTensor(torch.nn.Module):
    """A module to inject hashing debug output into a torch.Sequential or similar."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def forward(self, x: Tensor) -> Tensor:
        print(self.name, hash_tensor(x))
        return x


def hash_model(model: torch.nn.Module) -> str:
    return str(hash(tuple(hash_tensor(p.data, data_only=True) for p in model.state_dict().values())))[:8]


def explained_variance(y_pred: ArrayType, y_true: ArrayType) -> float:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        assert len(y_pred) == len(y_true)
        var_y = torch.var(y_true)  # type: ignore
        return np.nan if var_y == 0 else 1 - torch.var(y_true - y_pred).item() / var_y  # type: ignore
    elif isinstance(y_pred, np.ndarray):
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        assert len(y_pred) == len(y_true)
        var_y = np.var(y_true)
        return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred).item() / var_y  # type: ignore
    else:
        import tensorflow as tf

        if tf.is_tensor(y_pred):
            y_pred = tf.reshape(y_pred, [-1])
            y_true = tf.reshape(y_true, [-1])
            assert len(y_pred) == len(y_true)
            var_y = tf.math.reduce_variance(y_true)
            return np.nan if var_y == 0 else 1 - tf.math.reduce_variance(y_true - y_pred).numpy() / var_y
        else:
            raise ValueError


def create_action_storage(
    action_space: gym.spaces.Dict, batch_shape: tuple[int, ...], use_shared_memory: bool = False
) -> dict[str, Tensor]:
    action: dict[str, Tensor] = {}
    for k, v in action_space.items():
        if isinstance(v, OneHotMultiDiscrete):
            action[k] = torch.zeros(
                (*batch_shape, *assert_not_none(v.shape), v.max_categories),
                dtype=numpy_to_torch_dtype[assert_not_none(v.dtype).type],
            )
            if use_shared_memory:
                action[k] = action[k].share_memory_()
        elif isinstance(v, Box):
            action[k] = torch.zeros(
                (*batch_shape, *assert_not_none(v.shape)), dtype=numpy_to_torch_dtype[assert_not_none(v.dtype).type]
            )
            if use_shared_memory:
                action[k] = action[k].share_memory_()
        else:
            raise NotImplementedError
    return action


def create_observation_storage(
    observation_space: gym.spaces.Dict, batch_shape: tuple[int, ...], use_shared_memory: bool = False
) -> dict[str, Tensor]:
    observation: dict[str, Tensor] = {}
    for k, v in observation_space.items():
        observation[k] = torch.zeros(
            (*batch_shape, *assert_not_none(v.shape)),
            dtype=numpy_to_torch_dtype[v.dtype.type],
        )
        if use_shared_memory:
            observation[k] = observation[k].share_memory_()
    return observation


def copy(target: ArrayType, source: ArrayType) -> None:
    target[:] = source  # type: ignore


def masked_copy(target: ArrayType, source: ArrayType, mask: ArrayType) -> None:
    target[mask] = source[mask]  # type: ignore


def masked_copy_structure(target: ArrayType, source: ArrayType, mask: Optional[ArrayType]) -> None:
    """copy each atom of `source` into target, where `mask` is true"""
    if mask is not None:
        map_structure(partial(masked_copy, mask=mask), target, source)
    else:
        map_structure(copy, target, source)


def get_avalon_model_seed() -> Optional[int]:
    seed = os.getenv("AVALON_MODEL_SEED")
    return int(seed) if seed is not None else None


def seed_and_run_deterministically_if_enabled() -> None:
    if (seed := get_avalon_model_seed()) is not None:
        make_deterministic(seed)


def setup_new_process() -> None:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    torch.set_num_threads(1)
