# %%
from pathlib import Path

from common.imports import tqdm
from common.log_utils import enable_debug_logging
from contrib.utils import FILESYSTEM_ROOT
from datagen.generate_apks import generate_apk
from datagen.generate_apks import generate_apks
from datagen.generate_worlds import generate_worlds
from datagen.world_creation.constants import AvalonTask

enable_debug_logging()

# %%

RELEASE_BUILD = False
INCLUDE_WORLDS = True
PARALLEL_BUILD = False

if RELEASE_BUILD:
    APK_VERSION = ""
else:
    APK_VERSION = "debug"

assert APK_VERSION != "", "Run `git rev-parse HEAD` and set the APK version to be the most recent git hash."

print("APK_VERSION", APK_VERSION)

worlds_path = Path(f"{FILESYSTEM_ROOT}/avalon/worlds/{APK_VERSION}")
worlds_path.mkdir(exist_ok=True, parents=True)

# %%

tasks = [
    AvalonTask.EAT,
    AvalonTask.MOVE,
    AvalonTask.JUMP,
    AvalonTask.CLIMB,
    AvalonTask.DESCEND,
    AvalonTask.SCRAMBLE,
    AvalonTask.STACK,
    AvalonTask.BRIDGE,
    AvalonTask.PUSH,
    AvalonTask.THROW,
    AvalonTask.HUNT,
    AvalonTask.FIGHT,
    AvalonTask.AVOID,
    AvalonTask.EXPLORE,
    AvalonTask.OPEN,
    AvalonTask.CARRY,
    AvalonTask.NAVIGATE,
    AvalonTask.FIND,
    AvalonTask.SURVIVE,
    AvalonTask.GATHER,
]

# %%

if RELEASE_BUILD:
    practice_worlds = generate_worlds(
        base_output_path=worlds_path,
        tasks=tasks,
        num_worlds_per_task=2,
        start_seed=10000,
        is_practice=True,
        min_difficulty=0.5,
        is_recreating=True,
    )

# %%

if RELEASE_BUILD:
    actual_worlds = generate_worlds(
        base_output_path=worlds_path,
        tasks=tasks,
        num_worlds_per_task=50,
        start_seed=0,
        is_practice=False,
        min_difficulty=0.0,
        is_recreating=True,
    )

# %%

godot_path = Path("datagen/godot").absolute()

tmp_path = Path(f"/tmp/avalon/apks")
tmp_path.mkdir(exist_ok=True, parents=True)

output_path = Path(f"/opt/avalon/apks")
output_path.mkdir(exist_ok=True, parents=True)

apk_script = str(Path("scripts/apk.sh").absolute())

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

# %%

print("APK content hashes, excluding android/config.json:")
for output in output_apks:
    print(f"{output['apk_hash']} {output['apk_file']}")

# %%
