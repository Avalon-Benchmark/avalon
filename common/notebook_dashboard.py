from typing import Any
from typing import Dict

import attr
from ipywidgets import HBox


@attr.s(auto_attribs=True, collect_by_mro=True)
class BaseDashboard:
    outputs: Dict[str, HBox] = attr.ib(factory=dict)

    def layout(self):
        raise NotImplementedError()

    def get_sizes(self):
        return {}

    def update(self, name: str, value: Any):
        sizes = self.get_sizes()
        if name in sizes:
            value.height = sizes[name][0]
            value.width = sizes[name][1]
        self.outputs[name].children = (value,)


class InvisibleDashboard(BaseDashboard):
    def layout(self):
        pass

    def update(self, name: str, value: Any):
        pass


class LogDashboard(BaseDashboard):
    def log(self, data: Dict[str, float]):
        raise NotImplementedError
