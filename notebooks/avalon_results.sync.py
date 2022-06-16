# %%
import json
from collections import defaultdict
from typing import Dict
from typing import List

import numpy as np
from rliable import library as rly
from rliable import metrics

from agent.evaluation import get_latest_checkpoint_filename
from agent.evaluation import get_wandb_result_key
from agent.random.evaluation import get_random_result_key
from common.log_utils import logger
from contrib.s3_utils import SimpleS3Client
from datagen.world_creation.constants import TASKS_BY_TASK_GROUP
from datagen.world_creation.constants import AvalonTaskGroup

s3_client = SimpleS3Client()
data_key = "avalon_worlds__2f788115-ea32-4041-8cae-6e7cd33091b7.tar.gz"

# %%

## RANDOM BASELINE
random_result_keys = [get_random_result_key(i, data_key) for i in range(10)]
combined_random_results = defaultdict(list)
for result_key in random_result_keys:
    reported_data = json.loads(s3_client.load(result_key))
    for k, v in reported_data["all_results"].items():
        combined_random_results[k].append(round(v, ndigits=3))

random_agg_score = {}
for k, v in combined_random_results.items():
    random_agg_score[k] = np.mean(v)


# %%

## HUMAN BASELINE
human_scores_key = "avalon__human_scores__935781fe-267d-4dcd-9698-714cc891e985.json"

human_scores = json.loads(s3_client.load(human_scores_key))
human_agg_score = {}
folder_name_lookup = {}
for world_dir, human_results_on_world in human_scores.items():
    k = str(int(world_dir.split("__")[1]))
    folder_name_lookup[k] = world_dir
    human_agg_score[k] = np.mean(list(human_results_on_world.values()))

# %%
valid_levels = set(human_agg_score.keys()) & set(random_agg_score.keys())

human_fail_levels = set()
for k in valid_levels:
    if human_agg_score[k] <= random_agg_score[k]:
        human_fail_levels.add(k)
        fail_folder = folder_name_lookup[k]

valid_levels -= human_fail_levels
print("num levels", len(valid_levels))


def get_human_normalized_score(world_index: str, raw_score: float) -> float:
    if world_index not in valid_levels:
        logger.warning(f"{world_index} not in valid_levels returning unnormalized score!")
        return raw_score
    return max(0, raw_score - random_agg_score[world_index]) / (
        human_agg_score[world_index] - random_agg_score[world_index]
    )


world_id_by_task = defaultdict(list)
simple_tasks = set(x.value.lower() for x in TASKS_BY_TASK_GROUP[AvalonTaskGroup.SIMPLE])
compositional_tasks = set(x.value.lower() for x in TASKS_BY_TASK_GROUP[AvalonTaskGroup.COMPOSITIONAL])
for key in valid_levels:
    task = folder_name_lookup[key].split("__")[0]
    world_id_by_task[task].append(key)
    if task in simple_tasks:
        world_id_by_task["simple"].append(key)
    if task in compositional_tasks:
        world_id_by_task["compositional"].append(key)
    world_id_by_task["all"].append(key)


def task_dict_to_score_numpys(task_dict: Dict[str, List[float]]):
    np_task_dict: Dict[str, np.ndarray] = {}
    for task, keys in world_id_by_task.items():
        np_task_dict[task] = np.stack([np.array(task_dict[k]) for k in keys])
    return np_task_dict


PRINT_TASK_ORDER = [x.value.lower() for x in TASKS_BY_TASK_GROUP[AvalonTaskGroup.ALL]] + [
    "simple",
    "compositional",
    "all",
]


aggregate_func = lambda x: np.array(
    [
        metrics.aggregate_mean(x),
        metrics.aggregate_optimality_gap(x),
    ]
)


def print_aggregate_results(score_numpys: Dict[str, np.ndarray]):
    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(score_numpys, aggregate_func, reps=1000)
    for task in PRINT_TASK_ORDER:
        mean, opt_gap = aggregate_scores[task]
        mean_low, mean_high = aggregate_score_cis[task][:, 0]
        mean_sig = (mean_high - mean_low) / 2
        opt_gap_low, opt_gap_high = aggregate_score_cis[task][:, 1]
        opt_gap_sig = (mean_high - mean_low) / 2
        print(f"{task.upper()}: mean {mean:.3f} +- {mean_sig:.3f}, og: {opt_gap:.3f} +- {opt_gap_sig:.3f}")


# %%

## TORCHBEAST BASELINE
wandb_runs = [
    "sourceress/abe__torchbeast/15670wcf",
    "sourceress/abe__torchbeast/w0zt8c1r",
    "sourceress/abe__torchbeast/320exk7d",
    "sourceress/abe__torchbeast/20j4n9dd",
    "sourceress/abe__torchbeast/zq1hs2h3",
    "sourceress/abe__torchbeast/3szvt5xc",
    "sourceress/abe__torchbeast/so13nk61",
    "sourceress/abe__torchbeast/38z8lhh9",
]
combined_torchbeast_results = defaultdict(list)
raw_torchbeast_datas = []
for wandb_run in wandb_runs:
    checkpoint_filename = get_latest_checkpoint_filename(wandb_run, prefix="model_step_", suffix=".tar")
    logger.info(f"Loading results for {checkpoint_filename} from {wandb_run}")
    result_key = get_wandb_result_key(wandb_run, checkpoint_filename, data_key)
    reported_data = json.loads(s3_client.load(result_key))
    raw_torchbeast_datas.append(reported_data)
    missing_levels = valid_levels - set(reported_data["all_results"].keys())
    if len(missing_levels) > 0:
        logger.warning(f"{wandb_run} is missing {len(missing_levels)} levels, ignoring incomplete data")
    else:
        for k in valid_levels:
            level_data = reported_data["all_results"][k]
            combined_torchbeast_results[k].append(get_human_normalized_score(k, level_data))

torchbeast_score_numpys = task_dict_to_score_numpys(combined_torchbeast_results)

print("\nTorchbeast 50m aggregated scores")
print_aggregate_results(torchbeast_score_numpys)

# %%


## PPO BASELINE
wandb_runs = [
    "sourceress/abe__ppo/3jjf1kmx",
    "sourceress/abe__ppo/3laawj88",
    "sourceress/abe__ppo/2a4vx29a",
    "sourceress/abe__ppo/2iivkxvy",
    "sourceress/abe__ppo/2w50d2g3",
    "sourceress/abe__ppo/1v9ega3f",
]
checkpoint_filename = "model_29860800.pt"
combined_ppo_results = defaultdict(list)
raw_ppo_datas = []
for wandb_run in wandb_runs:
    logger.info(f"Loading results for {checkpoint_filename} from {wandb_run}")
    result_key = get_wandb_result_key(wandb_run, checkpoint_filename, data_key)
    reported_data = json.loads(s3_client.load(result_key))
    raw_ppo_datas.append(reported_data)
    missing_levels = valid_levels - set(reported_data["all_results"].keys())
    if len(missing_levels) > 0:
        logger.warning(f"{wandb_run} is missing {len(missing_levels)} levels, ignoring incomplete data")
    else:
        for k in valid_levels:
            level_data = reported_data["all_results"][k]
            combined_ppo_results[k].append(get_human_normalized_score(k, level_data))


ppo_score_numpys = task_dict_to_score_numpys(combined_ppo_results)
print("\nPPO 30m aggregated scores")
print_aggregate_results(ppo_score_numpys)

# %%

## TORCHBEAST CURRICULUM ABLATION
result_keys_by_curriculum = {
    "task_only": [
        "sourceress_abe__torchbeast_1viushv8__model_step_39388800.tar__avalon_worlds__2f788115-ea32-4041-8cae-6e7cd33091b7.tar.gz",
        "sourceress_abe__torchbeast_148f6z8f__model_step_43606400.tar__avalon_worlds__2f788115-ea32-4041-8cae-6e7cd33091b7.tar.gz",
        "sourceress_abe__torchbeast_t68l25tc__model_step_43913600.tar__avalon_worlds__2f788115-ea32-4041-8cae-6e7cd33091b7.tar.gz",
    ],
    "none": [
        "sourceress_abe__torchbeast_31n7esqp__model_step_20633600.tar__avalon_worlds__2f788115-ea32-4041-8cae-6e7cd33091b7.tar.gz",
        "sourceress_abe__torchbeast_1hnfpgr3__model_step_22275200.tar__avalon_worlds__2f788115-ea32-4041-8cae-6e7cd33091b7.tar.gz",
        "sourceress_abe__torchbeast_1bijqylb__model_step_16748800.tar__avalon_worlds__2f788115-ea32-4041-8cae-6e7cd33091b7.tar.gz",
    ],
    "meta_only": [
        "sourceress_abe__torchbeast_30l8m6t9__model_step_37008000.tar__avalon_worlds__2f788115-ea32-4041-8cae-6e7cd33091b7.tar.gz",
        "sourceress_abe__torchbeast_2sw328jm__model_step_37187200.tar__avalon_worlds__2f788115-ea32-4041-8cae-6e7cd33091b7.tar.gz",
        "sourceress_abe__torchbeast_rattuz9h__model_step_37552000.tar__avalon_worlds__2f788115-ea32-4041-8cae-6e7cd33091b7.tar.gz",
    ],
}
torchbeast_curriculum_results = {
    "task_only": defaultdict(list),
    "meta_only": defaultdict(list),
    "none": defaultdict(list),
}
for curriculum, result_keys in result_keys_by_curriculum.items():
    for result_key in result_keys:
        reported_data = json.loads(s3_client.load(result_key))
        missing_levels = valid_levels - set(reported_data["all_results"].keys())
        if len(missing_levels) > 0:
            logger.warning(f"{result_key} is missing {len(missing_levels)} levels, ignoring incomplete data")
        else:
            for k in valid_levels:
                level_data = reported_data["all_results"][k]
                torchbeast_curriculum_results[curriculum][k].append(get_human_normalized_score(k, level_data))


curriculum_score_numpys = {k: task_dict_to_score_numpys(v) for k, v in torchbeast_curriculum_results.items()}
print("\nCurriculum ablation aggregated scores")
for curriculum, scores in curriculum_score_numpys.items():
    print(f'\nCurriculum: {curriculum.replace("_", " ").capitalize()}')
    print_aggregate_results(scores)

# %%
