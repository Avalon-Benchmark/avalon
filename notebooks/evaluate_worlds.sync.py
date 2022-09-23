# %%
import shutil
from pathlib import Path

from avalon.common.log_utils import enable_debug_logging
from avalon.datagen.env_helper import display_video
from avalon.datagen.env_helper import visualize_worlds_in_folder
from avalon.datagen.generate_worlds import generate_worlds
from avalon.datagen.world_creation.constants import AvalonTask
from avalon.datagen.world_creation.types import DebugVisualizationConfig

enable_debug_logging()

NUM_WORLDS = 10
IS_GRAPH_MODE = False


tasks = [
    # AvalonTask.SURVIVE,
    # AvalonTask.THROW,
    # AvalonTask.CARRY,
    # needs to be tested:
    # AvalonTask.DESCEND,
    # AvalonTask.HUNT,
    # AvalonTask.FIGHT,
    # AvalonTask.AVOID,
    # AvalonTask.FIND,
    # AvalonTask.EXPLORE,
    # AvalonTask.SCRAMBLE,
    AvalonTask.NAVIGATE,
    # AvalonTask.GATHER,
    # AvalonTask.OPEN,
    # AvalonTask.EAT,
    # AvalonTask.MOVE,
    # AvalonTask.JUMP,
    # AvalonTask.CLIMB,
    # AvalonTask.STACK,
    # AvalonTask.BRIDGE,
    # AvalonTask.PUSH,
]


base_output_path = Path("/tmp/levels/visualize/")
if base_output_path.exists():
    shutil.rmtree(base_output_path)
base_output_path.mkdir(parents=True, exist_ok=True)

# %%

actual_worlds = generate_worlds(
    base_output_path=base_output_path,
    tasks=tasks,
    num_worlds_per_task=NUM_WORLDS,
    start_seed=0,
    is_practice=False,
    min_difficulty=0.0,
    is_recreating=True,
    debug_visualization_config=DebugVisualizationConfig(is_2d_graph_drawn=IS_GRAPH_MODE),
    is_async=not IS_GRAPH_MODE,
    is_generating_for_human=True,
)

# %%

NUM_ACTIONS = 20
RESOLUTION = 512

world_paths = [base_output_path / x.world_id for x in actual_worlds]
all_observations = visualize_worlds_in_folder(world_paths, RESOLUTION, NUM_ACTIONS)

for obs in all_observations:
    display_video(obs, size=(RESOLUTION, RESOLUTION))

# %%
