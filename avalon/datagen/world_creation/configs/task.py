import attr

from avalon.contrib.serialization import Serializable


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class TaskConfig(Serializable):
    pass


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class IndoorTaskConfig(TaskConfig):
    min_site_radius: float = 5.0
    max_site_radius: float = 12.0
