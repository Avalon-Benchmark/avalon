import pickle as pkl
from functools import partial
from pathlib import Path
from typing import Any
from typing import Dict

import numpy as np
import torch

from avalon.agent.common.types import Algorithm
from avalon.agent.common.util import postprocess_uint8_to_float
from avalon.contrib.utils import make_deterministic
from avalon.datagen.world_creation.tests.params import CANONICAL_SEED


def groom_observation_for_agent(
    observation: Dict[str, torch.Tensor], num_workers: int = 1, device: torch.device = torch.device("cpu")
) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        torch_obs = {
            k: torch.tensor(v).broadcast_to((num_workers, *v.shape)).to(device=device) for k, v in observation.items()
        }
        return postprocess_uint8_to_float(torch_obs)


def _dump(contents: Any, target_path: Path) -> None:
    with open(target_path, "wb") as target_file:
        pkl.dump(contents, target_file)


def _dump_dict_values(d: Dict[Any, Any], target_path: Path) -> None:
    # Using pickle.dump on numpy, torch and other objects can attach non-pertinent state to them which makes checking
    # deterministic behaviour hard. Use this function instead for such cases.
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            torch.save(v, target_path.parent / f"{target_path.name}_{k}.pt")
        elif isinstance(v, np.ndarray):
            np.save(str(target_path.parent / f"{target_path.name}_{k}"), v)
        else:
            pkl.dump(v, open(target_path.parent / f"{target_path.name}_{k}.pkl", "wb"))


def _load(source_path: Path) -> Any:
    with open(source_path, "rb") as source_file:
        return pkl.load(source_file)


def run_deterministic_forward_pass(
    agent: Algorithm, observation: Dict[str, torch.Tensor], seed: int = CANONICAL_SEED
) -> Dict[str, torch.Tensor]:
    assert agent.params.num_workers == 1, "Can't guarantee determinism when num_workers > 1"
    num_workers = 1

    ready_for_new_step = torch.ones((num_workers,), dtype=torch.bool)
    device = next(agent.parameters()).device
    torch_dones = torch.zeros((num_workers,), dtype=torch.bool, device=device)
    torch_obs = groom_observation_for_agent(observation, num_workers, device)
    make_deterministic(seed)
    agent.reset_state()
    agent.action_space.seed(seed)
    with torch.no_grad():
        action, _ = agent.rollout_step(torch_obs, torch_dones, ready_for_new_step, exploration_mode="eval")
    return action


assert_tensors_equal = partial(torch.testing.assert_close, rtol=0, atol=0)
