from pathlib import Path
from typing import Optional

import attr
import numpy as np

from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.constants import AvalonTask
from avalon.datagen.world_creation.worlds.compositional import CompositeTaskConfig
from avalon.datagen.world_creation.worlds.compositional import ForcedComposition
from avalon.datagen.world_creation.worlds.compositional import create_compositional_task
from avalon.datagen.world_creation.worlds.export import export_world


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class NavigateTaskConfig(CompositeTaskConfig):
    task: AvalonTask = AvalonTask.NAVIGATE


def generate_navigate_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
    task_config: NavigateTaskConfig = NavigateTaskConfig(),
    _FORCED: Optional[ForcedComposition] = None,
) -> None:
    world, locations = create_compositional_task(rand, difficulty, task_config, export_config, _FORCED=_FORCED)
    export_world(output_path, rand, world)
