# %%
import os
from pathlib import Path

from loguru import logger

from avalon.common.imports import tqdm
from avalon.common.log_utils import configure_local_logger
from avalon.contrib.utils import FILESYSTEM_ROOT
from avalon.contrib.utils import TEMP_DIR
from avalon.datagen.generate_apks import add_apk_version_to_server
from avalon.datagen.generate_apks import add_worlds_to_server
from avalon.datagen.generate_apks import generate_apk
from avalon.datagen.generate_apks import generate_apks
from avalon.datagen.generate_apks import upload_apk_to_server
from avalon.datagen.generate_worlds import generate_evaluation_worlds
from avalon.datagen.world_creation.constants import AvalonTask

configure_local_logger()

# %%

RELEASE_BUILD = False
INCLUDE_WORLDS = True
PARALLEL_BUILD = False

if RELEASE_BUILD:
    APK_VERSION = ""
else:
    APK_VERSION = "debug"

assert APK_VERSION != "", "Run `git rev-parse HEAD` and set the APK version to be the most recent git hash."

logger.info(f"{APK_VERSION=}")

worlds_path = Path(f"{FILESYSTEM_ROOT}/avalon/worlds/{APK_VERSION}")
worlds_path.mkdir(parents=True, exist_ok=True)

# %%

tasks = [
    AvalonTask.EAT,
    # AvalonTask.MOVE,
    # AvalonTask.JUMP,
    # AvalonTask.CLIMB,
    # AvalonTask.DESCEND,
    # AvalonTask.SCRAMBLE,
    # AvalonTask.STACK,
    # AvalonTask.BRIDGE,
    # AvalonTask.PUSH,
    # AvalonTask.THROW,
    # AvalonTask.HUNT,
    # AvalonTask.FIGHT,
    # AvalonTask.AVOID,
    # AvalonTask.EXPLORE,
    # AvalonTask.OPEN,
    # AvalonTask.CARRY,
    # AvalonTask.NAVIGATE,
    # AvalonTask.FIND,
    # AvalonTask.SURVIVE,
    # AvalonTask.GATHER,
]

# %%
practice_worlds = generate_evaluation_worlds(
    base_output_path=worlds_path,
    tasks=tasks,
    num_worlds_per_task=10,
    start_seed=10000,
    is_practice=True,
    min_difficulty=0.0,
    is_recreating=True,
    is_generating_for_human=True,
    num_workers=64,
)


# %%

if RELEASE_BUILD:
    actual_worlds = generate_evaluation_worlds(
        base_output_path=worlds_path,
        tasks=tasks,
        num_worlds_per_task=50,
        start_seed=0,
        is_practice=False,
        min_difficulty=0.0,
        is_recreating=True,
        is_generating_for_human=False,
        num_workers=64,
    )
else:
    actual_worlds = []

# %%

if RELEASE_BUILD:
    # makes an API request to the server to create new directories for the worlds
    #   NOTE: does not copy the worlds
    if len(practice_worlds) > 0:
        add_worlds_to_server(practice_worlds)
    if len(actual_worlds) > 0:
        add_worlds_to_server(actual_worlds)
    # make sure this apk version is valid
    add_apk_version_to_server(APK_VERSION)

# %%

godot_path = Path("avalon/datagen/godot")
tmp_path = Path(f"{TEMP_DIR}/avalon/apks")
output_path = Path(f"{FILESYSTEM_ROOT}/avalon/apks")
apk_script = f"{os.getcwd()}/scripts/apk.sh"

participant_ids = ["your_participant_ids_here"]

output_apks = []
if PARALLEL_BUILD:
    output_apks.extend(
        generate_apks(
            godot_path=godot_path,
            worlds_path=worlds_path,
            tmp_path=tmp_path,
            output_path=output_path,
            apk_script=apk_script,
            participant_ids=participant_ids,
            apk_version=APK_VERSION,
            include_worlds=INCLUDE_WORLDS,
            is_output_traced=False,
        )
    )
else:
    for participant_id in tqdm(participant_ids):
        out = generate_apk(
            godot_path=godot_path,
            worlds_path=worlds_path,
            tmp_path=tmp_path,
            output_path=output_path,
            apk_script=apk_script,
            participant_id=participant_id,
            apk_version=APK_VERSION,
            include_worlds=INCLUDE_WORLDS,
            is_output_traced=True,
        )
        output_apks.append(out)

logger.info("APK content hashes, excluding android/config.json:")
for output in output_apks:
    logger.info(f"{output['apk_hash']} {output['apk_file']}")

    if RELEASE_BUILD:
        upload_apk_to_server(output["apk_file"])

# %%
