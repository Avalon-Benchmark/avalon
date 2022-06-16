from pathlib import Path
from typing import Optional

import numpy as np

from datagen.world_creation.constants import AvalonTask
from datagen.world_creation.heightmap import ExportConfig
from datagen.world_creation.tasks.compositional import ForcedComposition
from datagen.world_creation.tasks.compositional import create_compositional_task
from datagen.world_creation.tasks.utils import TaskGenerationFunctionResult
from datagen.world_creation.tasks.utils import export_skill_world


def generate_navigate_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
    _FORCED: Optional[ForcedComposition] = None,
) -> TaskGenerationFunctionResult:
    world, locations = create_compositional_task(rand, difficulty, AvalonTask.NAVIGATE, export_config, _FORCED=_FORCED)
    export_skill_world(output_path, rand, world)

    # TODO tune hit points
    return TaskGenerationFunctionResult(starting_hit_points=1.0)
