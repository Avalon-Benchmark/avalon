import os
from typing import Dict
from typing import Generic
from typing import Sequence
from typing import Type
from typing import TypeVar
from typing import Union
from typing import cast

import attr
import wandb
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

from avalon.common.log_utils import get_experiment_name
from avalon.common.log_utils import get_log_path
from avalon.common.log_utils import logger
from avalon.contrib.serialization import Serializable
from avalon.contrib.serialization import flatten_dict
from avalon.contrib.serialization import inflate_dict
from avalon.contrib.utils import run_local_command

# note: you cannot start these keys with '_'!  wandb seems to trim those out
SYNCED_RANGE_PREFIX = "SYNCED_RANGE"
TYPE_KEY = "$_type"
REFERENCE_KEY = "reference"

T = TypeVar("T")
ParamType = TypeVar("ParamType", bound=Serializable)

GIT_HASH_KEY = "git_hash"
EXPERIMENT_NAME_KEY = "experiment_name"


@attr.s(auto_attribs=True, collect_by_mro=True, init=False)
class SyncedRange(Serializable, Generic[T]):
    value: T

    def __init__(self, value: T):
        self.value = value


SweepRangeType = Dict[str, Union[Sequence, SyncedRange]]


def init_wandb() -> Union[Run, RunDisabled, None]:
    log_path = get_log_path() / "wandb"
    return wandb.init(dir=str(log_path))


def to_wandb_config(params: Serializable) -> Dict:
    config_data = cast(Dict, flatten_dict(params.to_dict()))

    # add a key for the git hash as well
    git_hash_command_result = run_local_command("git rev-parse HEAD", is_checked=False)
    if git_hash_command_result.returncode == 0:
        git_hash = git_hash_command_result.stdout.decode("UTF-8").strip()
        assert len(git_hash) == 40
    elif "PYTEST_CURRENT_TEST" in os.environ:
        git_hash = "TESTIGITHASH"
    else:
        _git_hash_from_env = os.getenv("GIT_HASH")
        assert _git_hash_from_env is not None
        git_hash = _git_hash_from_env

    config_data[GIT_HASH_KEY] = git_hash
    config_data[EXPERIMENT_NAME_KEY] = get_experiment_name()
    return config_data


def get_params_from_wandb_config(wandb_config: Dict, klass: Type[ParamType]) -> ParamType:
    fixed_config = {}
    for key, value in wandb_config.items():
        # ignore keys that are the global synced range keys, and git_hash and experiment_name
        if not key.startswith(SYNCED_RANGE_PREFIX) and key != GIT_HASH_KEY and key != EXPERIMENT_NAME_KEY:
            if isinstance(value, dict) and value.get(TYPE_KEY) == SyncedRange.__name__:
                synced_range_name = value.get(REFERENCE_KEY)
                true_value = wandb_config[synced_range_name]
            else:
                true_value = value
            fixed_config[key] = true_value
    inflated_params_dict = inflate_dict(fixed_config)
    return klass.from_dict(inflated_params_dict)


def make_wandb_sweep_config(
    project_name: str,
    experiment_name: str,
    params: Serializable,
    sweep_ranges: SweepRangeType,
    method: str = "grid",
    description: str = "",
) -> Dict:
    config = to_wandb_config(params)
    logger.info(config)

    # check if sweep_ranges are actually applicable and fail if not
    valid_keys = set(config.keys())
    configured_keys = set(sweep_ranges.keys())
    invalid_keys = configured_keys.difference(valid_keys)
    if len(invalid_keys) > 0:
        raise Exception(f"Cannot run sweep, dictionary contains invalid keys: {invalid_keys}")

    # convert the synced ranges into global values instead
    synced_range_reference_by_id: Dict[int, dict] = {}
    normal_sweep_ranges = {}
    synced_sweep_ranges = {}
    global_sweep_ranges = {}
    for key, value in sweep_ranges.items():
        if isinstance(value, SyncedRange):
            if id(value) not in synced_range_reference_by_id:
                range_number = len(synced_range_reference_by_id)
                reference = f"{SYNCED_RANGE_PREFIX}_{range_number}"
                new_value = {TYPE_KEY: value.__class__.__name__, REFERENCE_KEY: reference}
                synced_range_reference_by_id[id(value)] = new_value
                global_sweep_ranges[reference] = value.value
            synced_sweep_ranges[key] = synced_range_reference_by_id[id(value)]
        else:
            normal_sweep_ranges[key] = value

    wandb_parameters = {
        **{k: {"value": v} for k, v in config.items()},
        **{k: {"values": v} for k, v in normal_sweep_ranges.items()},
        **{k: {"value": v} for k, v in synced_sweep_ranges.items()},
        **{k: {"values": v} for k, v in global_sweep_ranges.items()},
    }

    return {
        "name": experiment_name,
        "project": project_name,
        "description": description,
        "method": method,
        "parameters": wandb_parameters,
    }
