from typing import Type

from avalon.agent.common.params import Params
from avalon.agent.common.types import Algorithm
from avalon.agent.dreamer.params import DreamerParams
from avalon.agent.ppo.params import PPOParams
from avalon.contest.contest_params import ContestAlgorithmParams


def get_algorithm_cls(params: Params) -> Type[Algorithm]:
    """This is a bit hacky because we don't really want to import code from an algorithm we're not using."""
    algorithm_cls: Type[Algorithm]
    if issubclass(type(params), PPOParams):
        from avalon.agent.ppo.ppo import PPO

        algorithm_cls = PPO
    elif issubclass(type(params), DreamerParams):
        from avalon.agent.dreamer.dreamer import Dreamer

        algorithm_cls = Dreamer
    elif issubclass(type(params), ContestAlgorithmParams):
        from avalon.contest.eval import ContestAlgorithmWrapper

        algorithm_cls = ContestAlgorithmWrapper
    else:
        raise NotImplementedError(type(params).__name__)
    return algorithm_cls
