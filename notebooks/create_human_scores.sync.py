# %%
import gzip
import json
import shutil
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple

from common.imports import tqdm
from common.log_utils import enable_debug_logging
from contrib.s3_utils import SimpleS3Client
from contrib.utils import TEMP_DIR
from datagen.human_playback import DEFAULT_AVAILABLE_FEATURES
from datagen.human_playback import get_observations_from_human_recording

enable_debug_logging()


# %%

AVALON_BUCKET_NAME = "avalon-benchmark"
OBSERVATION_KEY = "avalon__all_observations__935781fe-267d-4dcd-9698-714cc891e985.tar.gz"

s3_client = SimpleS3Client(bucket_name=AVALON_BUCKET_NAME)

output_path = Path(f"{TEMP_DIR}/avalon")
output_path.mkdir(parents=True, exist_ok=True)

observation_path = output_path / "observation"
observation_path.mkdir(parents=True, exist_ok=True)

s3_client.download_to_file(key=OBSERVATION_KEY, output_path=output_path / OBSERVATION_KEY)
shutil.unpack_archive(output_path / OBSERVATION_KEY, observation_path, "gztar")

assert observation_path.exists()

# %%


def get_starting_hit_points(task: str) -> float:
    if task in ["survive", "find", "gather", "navigate"]:
        return 3.0
    elif task in ["stack", "carry", "explore"]:
        return 2.0
    else:
        return 1.0


def get_energy_cost_per_frame(starting_hit_points: float, task: str) -> float:
    if task in ["survive", "find", "gather", "navigate"]:
        return starting_hit_points / (15.0 * 60.0 * 10)
    elif task in ["stack", "carry", "explore"]:
        return starting_hit_points / (10.0 * 60.0 * 10)
    else:
        return starting_hit_points / (5.0 * 60.0 * 10)


def get_human_playback_files_for_world_id(human_playback_path: Path) -> Tuple[Path, Path, Path, Path]:
    paths = (
        human_playback_path / "actions.out",
        human_playback_path / "metadata.json",
        human_playback_path / "observations.out",
        human_playback_path / "human_inputs.out",
    )

    for raw_path in paths:
        path = Path(f"{raw_path}.gz")
        if path.exists():
            with gzip.open(str(path), "rb") as f_in:
                with open(raw_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

    return paths


def get_human_score_from_observation(
    world_id: str,
    user_id: str,
    user_path: Path,
    is_using_energy_expenditure: bool = False,
):
    task = world_id.split("__")[0]

    _, _, observations_path, _ = get_human_playback_files_for_world_id(human_playback_path=user_path)
    observations_path = user_path / "observations.out"

    if not observations_path.exists():
        path = Path(f"{observations_path}.gz")
        if path.exists():
            with gzip.open(str(path), "rb") as f_in:
                with open(observations_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            return dict(world_id=world_id, user_id=user_id, score=0.0, is_error=True, is_reset=False)

    is_reset = (user_path / "reset.marker").exists()

    human_observations = get_observations_from_human_recording(
        observations_path=str(observations_path),
        available_features=DEFAULT_AVAILABLE_FEATURES,
    )

    hit_points = get_starting_hit_points(task)
    energy_cost = get_energy_cost_per_frame(hit_points, task)

    # NOTE: not using energy expenditure because humans spend a lot of energy on long levels
    total_energy_coefficient = 1e-4
    body_kinetic_energy_coefficient = 0.0
    body_potential_energy_coefficient = 0.0
    head_potential_energy_coefficient = 0.0
    left_hand_kinetic_energy_coefficient = 0.0
    left_hand_potential_energy_coefficient = 0.0
    right_hand_kinetic_energy_coefficient = 0.0
    right_hand_potential_energy_coefficient = 0.0

    # skip the first frame because there are extremely high energy costs
    for obs in human_observations[1:]:
        total_energy_expenditure = (
            body_kinetic_energy_coefficient * obs.physical_body_kinetic_energy_expenditure.item()
            + body_potential_energy_coefficient * obs.physical_body_potential_energy_expenditure.item()
            + head_potential_energy_coefficient * obs.physical_head_potential_energy_expenditure.item()
            + left_hand_kinetic_energy_coefficient * obs.physical_left_hand_kinetic_energy_expenditure.item()
            + left_hand_potential_energy_coefficient * obs.physical_left_hand_potential_energy_expenditure.item()
            + right_hand_kinetic_energy_coefficient * obs.physical_right_hand_kinetic_energy_expenditure.item()
            + right_hand_potential_energy_coefficient * obs.physical_right_hand_potential_energy_expenditure.item()
        )
        total_energy_expenditure *= total_energy_coefficient

        hit_points += obs.reward.item() - energy_cost

        if is_using_energy_expenditure:
            hit_points -= total_energy_expenditure

        if obs.is_dead:
            return dict(world_id=world_id, user_id=user_id, score=0.0, is_error=False, is_reset=is_reset)
        if obs.is_food_present_in_world < 0.1:
            return dict(world_id=world_id, user_id=user_id, score=hit_points, is_error=False, is_reset=is_reset)

    return dict(world_id=world_id, user_id=user_id, score=max(0.0, hit_points), is_error=False, is_reset=is_reset)


# %%

score_by_world_id = defaultdict(dict)
resets_by_user_id = defaultdict(list)
all_errors = []


def on_done(result):
    if not result.get("is_error"):
        world_id = result["world_id"]
        user_id = result["user_id"]
        score = result["score"]
        score_by_world_id[world_id][user_id] = score
        if result.get("is_reset"):
            resets_by_user_id[user_id].append(world_id)


def on_error(error: BaseException):
    print("Evaluation failed!")
    all_errors.append(error)
    raise error


num_processes = 20

results = []

with Pool(processes=num_processes) as worker_pool:
    requests = []
    for world_path in list(observation_path.iterdir()):
        world_id = world_path.name
        if (world_path / "ignored.marker").exists() or world_id.startswith("practice"):
            continue
        for user_path in world_path.iterdir():
            user_id = user_path.name
            if (user_path / "crash").exists():
                continue

            task_name, seed, difficulty = world_id.split("__")
            cleaned_world_id = f"{task_name}__{int(seed)}__{difficulty}"

            request = worker_pool.apply_async(
                get_human_score_from_observation,
                kwds={
                    "world_id": cleaned_world_id,
                    "user_id": user_id,
                    "user_path": user_path,
                },
                callback=on_done,
                error_callback=on_error,
            )
            requests.append(request)
    for request in tqdm(requests):
        request.wait()
        if request._success:
            results.append(request.get())
    worker_pool.close()
    worker_pool.join()

# %%
