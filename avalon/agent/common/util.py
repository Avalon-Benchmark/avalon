import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import attr
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from avalon.agent.common import wandb_lib
from avalon.contrib.utils import make_deterministic


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


def pack_1d_list(sequence: List, out_cls: Type):
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


def pack_2d_list(batch: List[List], out_cls: Type):
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


def postprocess_uint8_to_float(data: Dict[str, torch.Tensor], observation_prefix: str = "") -> Dict[str, torch.Tensor]:
    """Convert uint8 (0,255) to float (-.5, .5) in a dictionary of rollout data.

    We use this to keep images as uint8 in storage + transfer.
    If observation_prefix is passed, then only keys with that prefix will be looked at for transforming.
    This allows handling a whole batch of rollout data (with actions, values, etc), which is convenient.
    """
    out = {}
    for k, v in data.items():
        if k.startswith(observation_prefix) and v.dtype in (np.uint8, torch.uint8):
            v = v / 255.0 - 0.5
        out[k] = v
    return out


ArrayType = Union[NDArray, torch.Tensor]


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


def hash_tensor(tensor: torch.Tensor) -> int:
    return hash(tuple(tensor.cpu().contiguous().view(-1).tolist()))


def hash_model(model: torch.nn.Module) -> int:
    return hash(tuple(hash_tensor(p.data) for p in model.state_dict().values()))


def get_avalon_model_seed() -> Optional[int]:
    seed = os.getenv("AVALON_MODEL_SEED")
    return int(seed) if seed is not None else None


def seed_and_run_deterministically_if_enabled() -> None:
    if (seed := get_avalon_model_seed()) is not None:
        make_deterministic(seed)
