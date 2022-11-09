import attr

from avalon.agent.ppo.params import OnPolicyParams


@attr.s(auto_attribs=True, frozen=True)
class ContestAlgorithmParams(OnPolicyParams):
    pass
