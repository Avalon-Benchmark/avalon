# %%
import gzip
import json
import shutil
import tarfile
from collections import OrderedDict
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict
from typing import List
from typing import NamedTuple

from avalon.common.imports import tqdm
from avalon.common.log_utils import enable_debug_logging
from avalon.common.log_utils import logger
from avalon.contrib.s3_utils import SimpleS3Client
from avalon.contrib.utils import FILESYSTEM_ROOT
from avalon.datagen.env_helper import create_env
from avalon.datagen.env_helper import get_action_type_from_config
from avalon.datagen.godot_env.goals import AvalonGoalEvaluator
from avalon.datagen.godot_env.goals import GoalProgressResult
from avalon.datagen.godot_env.observations import AvalonObservation
from avalon.datagen.human_playback import get_observations_from_human_recording
from avalon.datagen.human_playback import get_oculus_playback_config
from avalon.datagen.world_creation.constants import AvalonTask
from avalon.datagen.world_creation.world_generator import GenerateAvalonWorldParams

enable_debug_logging()

# %%


class ScoreResult(NamedTuple):
    world_id: str
    user_id: str
    score: float
    is_error: bool
    is_reset: bool


class HumanScores(NamedTuple):
    score_by_world_id: Dict[str, Dict[str, float]]
    resets_by_user_id: Dict[str, List[str]]
    uncaught_errors: List[BaseException]
    expected_errors: List[ScoreResult]


class InvalidEpisode(Exception):
    pass


class PathDoesNotExit(Exception):
    pass


class UnexpectedPath(Exception):
    pass


VALID_APK_VERSIONS = [
    "6a88384c83e5a103cb2a10d4561315297d5019d2",
    "974025deded7ebe9c39d95d472048ec267d6caad",
    "d217981d161e790ac702b46ecfa8286bea153d54",
    "df9b06e74922efb57bc582931588d08e015f5036",
]


def get_human_score(
    world_params: GenerateAvalonWorldParams, observations: List[AvalonObservation]
) -> GoalProgressResult:
    goal_evaluator = AvalonGoalEvaluator()
    goal_evaluator.reset(observations[0], world_params)
    for obs in observations[1:]:
        progress = goal_evaluator.calculate_goal_progress(obs)
        if progress.is_done:
            return progress

    raise InvalidEpisode("is_done flag never set to True")


def _read_gzip_path_if_path_does_not_exist(path: Path) -> Path:
    gzip_path = Path(f"{path}.gz")
    if gzip_path.exists() and not path.exists():
        with gzip.open(str(gzip_path), "rb") as f_in:
            with open(path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    return path


def get_human_score_from_data(
    user_path: Path, world_id: str, user_id: str, selected_features: OrderedDict
) -> ScoreResult:
    task, seed, difficulty = world_id.split("__")

    is_reset = (user_path / "reset.marker").exists()

    if is_reset:
        return ScoreResult(
            world_id=world_id,
            user_id=user_id,
            score=0.0,
            is_error=False,
            is_reset=True,
        )
    user_path_with_apk_versions = [user_path / x for x in VALID_APK_VERSIONS if (user_path / x).exists()]
    if len(user_path_with_apk_versions) > 1:
        raise UnexpectedPath(f"Found multiple paths {user_path_with_apk_versions}")

    if len(user_path_with_apk_versions) == 0:
        logger.error(
            f"No sub-path in {user_path} that matches {VALID_APK_VERSIONS}. Either run just started or something went wrong."
        )
        return ScoreResult(world_id=world_id, user_id=user_id, score=0.0, is_error=True, is_reset=False)

    user_path_with_apk_version = user_path_with_apk_versions[0]

    observations_path = _read_gzip_path_if_path_does_not_exist(user_path_with_apk_version / "observations.out")
    if not observations_path.exists():
        raise PathDoesNotExit(str(observations_path))

    human_observations = get_observations_from_human_recording(
        observations_path=observations_path,
        selected_features=selected_features,
    )

    world_params = GenerateAvalonWorldParams(
        task=AvalonTask[task.upper()], difficulty=float(difficulty), seed=int(seed), index=0, output=""
    )
    try:
        progress = get_human_score(world_params, human_observations)
    except InvalidEpisode as e:
        logger.error(e)
        return ScoreResult(world_id=world_id, user_id=user_id, score=0.0, is_error=True, is_reset=False)

    return ScoreResult(
        world_id=world_id,
        user_id=user_id,
        score=progress.log["score"],
        is_error=False,
        is_reset=False,
    )


def get_all_human_scores(root_path: Path) -> HumanScores:
    score_by_world_id: Dict[str, Dict[str, float]] = defaultdict(dict)
    resets_by_user_id: Dict[str, List[str]] = defaultdict(list)
    uncaught_errors: List[BaseException] = []
    expected_errors: List[ScoreResult] = []

    def on_done(result: ScoreResult):
        if result.is_error:
            expected_errors.append(result)
        elif result.is_reset:
            resets_by_user_id[result.user_id].append(result.world_id)
        else:
            score_by_world_id[result.world_id][result.user_id] = result.score

    def on_error(error: BaseException):
        logger.error("Evaluation failed!")
        uncaught_errors.append(error)
        raise error

    num_processes = 20

    pool_results = []

    config = get_oculus_playback_config(is_using_human_input=False)
    action_type = get_action_type_from_config(config)
    env = create_env(config, action_type)
    selected_features = env.observation_context.selected_features
    env.close()

    with Pool(processes=num_processes) as worker_pool:
        requests = []
        for world_path in list(root_path.iterdir()):
            world_id = world_path.name
            if (
                (world_path / "ignored.marker").exists()
                or world_id.startswith("practice")
                or world_id.startswith("worlds")
                or world_id.startswith("versions")
            ):
                continue
            for user_path in world_path.iterdir():
                user_id = user_path.name
                if (user_path / "crash").exists():
                    continue

                task_name, seed, difficulty = world_id.split("__")
                cleaned_world_id = f"{task_name}__{int(seed)}__{difficulty}"

                request = worker_pool.apply_async(
                    get_human_score_from_data,
                    kwds={
                        "user_path": user_path,
                        "world_id": cleaned_world_id,
                        "user_id": user_id,
                        "selected_features": selected_features,
                    },
                    callback=on_done,
                    error_callback=on_error,
                )
                requests.append(request)
        for request in tqdm(requests):
            request.wait()
            if request._success:
                pool_results.append(request.get())
        worker_pool.close()
        worker_pool.join()

    return HumanScores(
        score_by_world_id,
        resets_by_user_id,
        uncaught_errors,
        expected_errors,
    )


# %%

AVALON_BUCKET_NAME = "avalon-benchmark"
s3_client = SimpleS3Client(bucket_name=AVALON_BUCKET_NAME)

tmp_path = Path(f"{FILESYSTEM_ROOT}/tmp/")
key = "avalon_human_data__0908.tar.gz"
tar_path = tmp_path / key
s3_client.download_to_file(key=key, output_path=tar_path)

# %%

root_path = tmp_path / "avalon_human_data"
tar = tarfile.open(tar_path)
tar.extractall(path=root_path.parent)
tar.close()
tar_path.unlink()

# %%

results = get_all_human_scores(root_path)

# %%

json.dump(results.score_by_world_id, open(tmp_path / "human_scores.json", "w"))
