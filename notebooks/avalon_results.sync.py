"""
This file generates the plots in our paper for human and agent performance in Avalon.
"""
# %%
import json
from collections import defaultdict
from typing import Dict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.colors import hsv_to_rgb
from rliable import library as rly
from rliable import metrics
from tabulate import tabulate

from avalon.common.log_utils import logger
from avalon.contrib.s3_utils import SimpleS3Client
from avalon.datagen.world_creation.constants import TASKS_BY_TASK_GROUP
from avalon.datagen.world_creation.constants import AvalonTaskGroup

s3_client = SimpleS3Client()
FIXED_WORLD_KEY = "avalon_worlds__0824_full.tar.gz"


# to upload
# data_key = "avalon_worlds__d24d1746-4b0d-43da-bc83-338bc1eab441.tar.gz"
# s3_client.upload_from_file(Path("/tmp/avalon_worlds/d24d1746-4b0d-43da-bc83-338bc1eab441.tar.gz"), data_key)

# %%

## RANDOM BASELINE

random_result_keys = [f"random_policy__{i}__{FIXED_WORLD_KEY}" for i in range(10)]
combined_random_results = defaultdict(list)
for result_key in random_result_keys:
    reported_data = json.loads(s3_client.load(result_key))
    for k, v in reported_data["all_results"].items():
        combined_random_results[k].append(v)

random_agg_score = {}
for k, v in combined_random_results.items():
    random_agg_score[k] = np.mean(v)

# %%

## HUMAN BASELINE
human_scores_key = "avalon_human_scores__0826"

# to upload:
# with open("/mnt/private/avalon/human_scores__0824.json", "r") as f:
#     human_scores = json.load(f)
# s3_client.save(human_scores_key, json.dumps(human_scores))

human_scores = json.loads(s3_client.load(human_scores_key))
human_agg_score = {}
folder_name_lookup = {}
for world_dir, human_results_on_world in human_scores.items():
    k = str(int(world_dir.split("__")[1]))
    folder_name_lookup[k] = world_dir
    human_agg_score[k] = np.mean(list(human_results_on_world.values()))

# %%

manually_excluded = set()
valid_levels = set(human_agg_score.keys())  # & set(random_agg_score.keys())
human_fail_levels = set()
for k in valid_levels:
    if human_agg_score[k] <= random_agg_score[k]:
        human_fail_levels.add(k)
        fail_folder = folder_name_lookup[k]
        logger.warning(f"No human succeeded at level {fail_folder}! {human_scores[fail_folder]}")

valid_levels -= human_fail_levels
valid_levels -= manually_excluded


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
        if task not in aggregate_scores:
            continue
        mean, opt_gap = aggregate_scores[task]
        mean_low, mean_high = aggregate_score_cis[task][:, 0]
        mean_sig = (mean_high - mean_low) / 2
        opt_gap_low, opt_gap_high = aggregate_score_cis[task][:, 1]
        opt_gap_sig = (mean_high - mean_low) / 2
        logger.info(f"{task.upper()}: mean {mean:.3f} +- {mean_sig:.3f}, og: {opt_gap:.3f} +- {opt_gap_sig:.3f}")


def latex_aggregate_results(score_numpys: Dict[str, np.ndarray]) -> Dict[str, Dict[str, str]]:
    lines = {}
    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(score_numpys, aggregate_func, reps=1000)
    for task in PRINT_TASK_ORDER:
        if task not in aggregate_scores:
            lines[task] = {
                "mean": "-",
                "optimality_gap": "-",
            }
            continue
        mean, opt_gap = aggregate_scores[task]
        mean_low, mean_high = aggregate_score_cis[task][:, 0]
        mean_sig = (mean_high - mean_low) / 2
        opt_gap_low, opt_gap_high = aggregate_score_cis[task][:, 1]
        opt_gap_sig = (mean_high - mean_low) / 2
        lines[task] = {
            "mean": f"{mean:.3f} $\pm$ {mean_sig:.3f}",
            "optimality_gap": f"{opt_gap:.3f} $\pm$ {opt_gap_sig:.3f}",
        }
    return lines


# %%


FINISHED_RUNS = {
    "single_task_navigate": [
        "d2e999ac-539c-480e-9286-843e52152f93",
        "d141c933-1120-4a65-b759-2e2bcf46240d",
        "98802cc8-8a41-41b9-9a93-2a8545bca7cb",
        "7a37d446-5391-47c9-9742-2fd679b17b1a",
        "46b9dd6e-5d6e-4b1d-a500-cc70b2afac02",
        "5b1c2b4a-d43a-4680-bba7-a7be4f2d8915",
    ],
    "single_task_carry": [
        "8222027c-49b2-4950-aa14-6a4d612f2f70",
        "95fe5e36-729e-462e-9902-5305db492d15",
        "f04bda8e-e940-4edc-9e70-bbfd7e1d7253",
        "d78e3ae2-c978-42ac-b7d6-0a3c75287452",
        "9970abc0-9a50-46d7-bd42-5ab7fe8a23c0",
        "1ea2118a-7418-4c99-8c25-919e3eeb20a2",
    ],
    "single_task_bridge": [
        "02e3b730-7f87-4f93-be01-d77d12adba04",
        "42dac847-fc97-4f6e-a111-6cd67b06ff0e",
        "f7e94e54-43be-4f7a-bd49-d8d8ccd3c15f",
        "b506fb63-897e-491c-ad3f-0199cec3fc50",
        "3cb5f25b-1b1c-4e6d-9d34-c17966954af3",
    ],
    "single_task_stack": [
        "4f63e2d0-4a17-4911-a819-20ea2c77c8f4",
        "d56aea61-4077-4574-a6ff-c75340825d10",
        "008819b4-fc99-41e8-80c2-1a3571875053",
        "b3b92641-e981-4133-89b3-0cb075c7b141",
        "3e587d03-18ae-4131-ba0a-e680cbb0924b",
    ],
    "single_task_open": [
        "2d748ad1-f250-455a-b5a8-5ddfbf1537d7",
        "f730db29-fadb-4d04-9ebf-62f8e4e5e271",
        "1e771c08-6a2a-4ed8-abe7-c372a3875710",
        "131fb969-7262-4f36-860d-61122016eefa",
        "18f30685-5c54-4bc1-903f-844bac12187d",
        "d4399557-1e4d-4452-9436-11a8c2fabc05",
    ],
    "single_task_push": [
        "528b6372-9470-4859-8f52-d66a3b1ec986",
        "15a9f45a-319e-4254-a892-975c98bfbc95",
        "d462b56e-5228-427d-93c3-e347fd8154f5",
        "8e8870cd-cfa3-483e-b93d-fa089ab7e51b",
        "5f19829d-f0e6-4c25-b5b2-03d6eb1f701d",
    ],
    "single_task_fight": [
        "f3ff80c8-fbc3-4277-bdde-364ded17407f",
        "26edb4a1-d4f0-47c8-97f5-f3bba1344f67",
        "f19628c2-2084-41dc-8618-952a137f28df",
        "da2aa3aa-6361-4c32-99f5-926c6891cea9",
        "b5599519-8e81-479b-a670-1c9059f2db1f",
    ],
    "single_task_hunt": [
        "fb44000f-b1ec-48a8-8c60-9a332cb64f06",
        "b7a0e8a9-4caa-4b4d-8339-3c5c3d012893",
        "42144a8b-baa4-44f0-8752-0aeecaaeb868",
        "990bc5b7-5895-4c70-8ef8-70ce28a82a1d",
        "b96c9bbe-ade4-4ea0-bd96-2437b77daa24",
        "e4b87d99-7187-44bd-a636-d7900705b71b",
    ],
    "single_task_avoid": [
        "05e7cccc-cfc9-4345-a5f2-ea43f534a289",
        "8e65b732-2d04-4075-a714-9442af3e736b",
        "e8b864c9-6ac4-4a7d-99b2-fa85cc2b3010",
        "9b7ebfb4-6cf4-4fa9-a373-3cbc118389b0",
        "edf5e5d7-2fd0-43a5-a1cc-0276287af992",
        "556771f1-0579-4a64-a8be-ffcc1bd2b1c2",
    ],
    "single_task_throw": [
        "79c194f7-569d-40dd-9c82-1703ff799cfd",
        "6954376e-b840-4053-98ac-89336c4174bb",
        "9ec8cc8f-d318-45ac-ba8d-cc7d6c1b801c",
        "0bf62354-33e2-48df-96a7-2c2a1f73fedf",
        "e9ef062a-cd64-445e-b086-b15c6f33b346",
        "18e1eb11-e44d-457e-a7a8-2522ede2871e",
    ],
    "single_task_descend": [
        "a663d1a4-0e03-423a-a8db-1a5675b7d0d0",
        "1bce0f94-9924-4c30-8e0f-ee966e14ea34",
        "94e63f45-a7fb-4dce-beb2-01a97af2e0e6",
        "0cc8a42a-e73e-4c88-bd94-6a2d4eaa46f5",
        "0d915730-3795-4703-8834-01b74ff37f72",
        "9cfeb379-a59f-42cd-93df-2368fe96903b",
    ],
    "no_curr": [
        "2182d660-339d-4b59-a78a-f36b8be591d9",
        "9756bca1-a589-4644-a416-cc234904bc78",
        "3a610c73-5d22-4582-a9f8-6444dfcc4feb",
        "a18738ad-0fc4-4629-8e9e-5d6170dadd56",
        "a4932faa-638e-420e-bf2a-0f79b2cc52de",
        "f90b498a-15fe-40e8-8606-66ab8d10a445",
    ],
    "single_task_climb": [
        "076bfe0d-48ae-4033-ae28-c796a28bda51",
        "f13797b0-fa33-4d52-a8ab-170e9ceb1950",
        "47c7b009-f583-45b3-b104-dcb64e427129",
        "2c38e712-d32c-4a1f-9cbb-02560f911c49",
        "d08f208e-b237-4d0f-a38b-8dcd589226df",
        "e0097c76-332a-463c-8e17-718040eed51c",
    ],
    "single_task_scramble": [
        "39f106fe-a507-4545-9016-a72173328da2",
        "333f76e4-8c9a-4f82-bf73-5a7992b7bba7",
        "82b39ad2-3c14-4151-b8d0-a464640db8e8",
        "598acb50-4674-4dfb-8fb6-3c5ec2be5d75",
        "e0f267ce-4021-458e-b41d-0b81da1790bb",
        "cf9a1fad-9cd9-4e6e-b462-1955e17a84d8",
    ],
    "single_task_explore": [
        "eeb7ecbb-5303-4922-a236-35a86c8aa2eb",
        "9ebb3ac6-40e2-4a79-8282-a6e9f4e7ec9d",
        # "da1b4473-b78c-4891-bb57-7f52d81ae395",
        "d042966c-f793-487b-9a2e-35e121520ddb",
        "00bc5be8-6fbc-4802-821d-f5c6bb71119b",
        "3c300c1b-2f7b-4e2e-b959-56655ec31ea1",
    ],
    "single_task_jump": [
        "cf6ed141-ba00-4b8d-ae9c-8fe6217ed937",
        "e15ba89e-fd78-465a-b47f-9d76226ffb5c",
        "cf1efce4-f2f4-46f1-a2fa-c032868543e5",
        "19c8d3ab-7e08-485b-8950-b323b8f0f4d6",
        "168f5d1e-cb76-40f7-97a9-e2866c475038",
        "b477163c-32c6-4d7b-9551-dbe626fba751",
    ],
    "single_task_move": [
        "27d04e34-db0c-4d03-a5a0-1aa0d75df6a7",
        "067d3abd-0388-4421-98c7-419014b8360d",
        "45f62ceb-e0ff-4cc7-97a6-c75bdc0fa44e",
        "be7f951b-6645-4b38-8ab9-b7c9fc207ff8",
        "717909ba-9b05-4ade-8358-b6686efad897",
        "e8d2bf33-ec8d-4a1b-a6c3-63342330507e",
    ],
    "single_task_eat": [
        "295908c0-749c-400b-b207-2bdf335f0d13",
        "461591f5-6da3-4314-a275-76f3806074a6",
        "2eccc6b2-c348-4c5a-b585-d87ed7c99e85",
        "4771fb18-5a48-415a-933a-d76f60ca9dd9",
        "91729854-adfc-4630-96ff-60a7eb1d0f0c",
        "ce1ca135-59f5-446c-911d-1047113e903f",
    ],
    "multi_task_compositional": [
        "299772bd-5607-42d4-9b54-c497b420dfa4",
        "7be73111-92a7-4a5d-8b06-4dd1381c6823",
        "0ef0fb63-cbd3-460c-be59-a724c4c54ee1",
        "76afa191-45f1-43dc-a821-717ce215a1fe",
        "5f0dacb8-9067-4192-80a6-e34ff047dcd8",
        "8050a696-2ee8-485e-a1f6-206f684baf8e",
    ],
    "multi_task_basic": [
        "00b11b8a-7e74-4ebd-87e6-82ae500ca2ab",
        "f99fecba-9827-4b10-9569-bde0f98b7cca",
        "766165dd-57a5-4db4-a1ed-109d7229a832",
        "6d6a7a81-36a4-4916-9e21-997a843dbd0c",
        "70dbe576-71f8-494b-a820-b3aafd0480c5",
        "f4c7614f-c973-4c3b-baa8-7a7c532fbc74",
    ],
    "multi_task_all": [
        "9c225872-d04d-420c-b94e-1df06c49aa62",
        "d0bfd187-28ac-4dc5-ae83-2af3ef605aef",
        "ec2ce66c-be85-4ebc-b69d-55569c72e168",
        "c896839d-abc7-4b90-8a5f-3104ea554842",
        "65e7233f-08d2-4816-8549-92eaf58d8337",
        "f5a38e34-3ba4-4e3d-9615-b920e8f087d8",
    ],
    "single_task_survive": [
        "a3e47df0-eee7-499a-823a-3cada24aa2d3",
        "a5e8f34f-61dc-4db8-b811-f2e8617d60d2",
        "e1722229-9ceb-4ef7-9a2d-8bbbf81f0b4c",
        "f31efa63-00e5-4a6c-803e-9ebd09096801",
        "7212c774-e38b-4f92-91c3-039b0573f9ac",
    ],
    "single_task_gather": [
        "3a741d47-3aed-4a7c-89b7-c48a637e0cc8",
        "51ab7b25-926c-4153-9393-cce90a41a26c",
        "fc8fd934-dd60-4f34-9b38-fb5010934422",
        "d6dca059-eecf-499a-91dd-12ce6633449b",
        "a1e3c387-4fda-4e11-a7bb-b8dec1cc7cf0",
    ],
    "single_task_find": [
        "9695f3fa-4439-419a-b27f-48e45c4c7bcf",
        "5d42fe7d-678c-4ad7-a2a6-c814c45f2783",
        "c4bf76e8-b50e-45f9-9036-ceb7859d568c",
        "619aa9cd-da54-4e93-8904-963b4ef21a67",
        "81119314-2e65-411c-9b29-ec586c774574",
    ],
    "500m": [
        "42c4f566-49d1-45c6-a4f2-1185ac49238c",
        "cdb06a43-4dfb-4d52-878d-8495075a4a38",
        "bd10b76d-ce5b-4e0a-8514-47d75e777ae6",
    ],
}


def get_wandb_result_key(xpid: str, fixed_world_key: str, checkpoint: str = "final"):
    return f"avalon_eval__{xpid}__{fixed_world_key}__{checkpoint}"


# %%

## TORCHBEAST BASELINE

wandb_runs = FINISHED_RUNS["multi_task_basic"]
combined_torchbeast_results = defaultdict(list)
raw_torchbeast_datas = []
for uuid in wandb_runs:
    result_key = get_wandb_result_key(uuid, FIXED_WORLD_KEY)
    reported_data = json.loads(s3_client.load(result_key))
    raw_torchbeast_datas.append(reported_data)
    missing_levels = valid_levels - set(reported_data["all_results"].keys())
    if len(missing_levels) > 0:
        logger.warning(f"{uuid} is missing {len(missing_levels)} levels, ignoring incomplete data")
    else:
        for k in valid_levels:
            level_data = reported_data["all_results"][k]
            combined_torchbeast_results[k].append(get_human_normalized_score(k, level_data))

torchbeast_score_numpys = task_dict_to_score_numpys(combined_torchbeast_results)

logger.info("\nTorchbeast 50m aggregated scores")
print_aggregate_results(torchbeast_score_numpys)

# %%


## PPO BASELINE
result_keys = [
    "avalon_eval__zack_zack_ppo_3t0oz4qc_final.pt__avalon_worlds__0824_full.tar.gz__final",
    "avalon_eval__zack_zack_ppo_3ox73b4a_final.pt__avalon_worlds__0824_full.tar.gz__final",
    "avalon_eval__zack_zack_ppo_fhgppehq_final.pt__avalon_worlds__0824_full.tar.gz__final",
    "avalon_eval__zack_zack_ppo_1acn5jsy_final.pt__avalon_worlds__0824_full.tar.gz__final",
    "avalon_eval__zack_zack_ppo_1wli6rk8_final.pt__avalon_worlds__0824_full.tar.gz__final",
]
combined_ppo_results = defaultdict(list)
raw_ppo_datas = []
for result_key in result_keys:
    reported_data = json.loads(s3_client.load(result_key))
    raw_ppo_datas.append(reported_data)
    missing_levels = valid_levels - set(reported_data["all_results"].keys())
    if len(missing_levels) > 0:
        logger.warning(f"{result_key} is missing {len(missing_levels)} levels, ignoring incomplete data")
    else:
        for k in valid_levels:
            level_data = reported_data["all_results"][k]
            combined_ppo_results[k].append(get_human_normalized_score(k, level_data))


ppo_score_numpys = task_dict_to_score_numpys(combined_ppo_results)
logger.info("\nPPO 50m aggregated scores")
print_aggregate_results(ppo_score_numpys)

# %%


## Dreamer BASELINE
result_keys = [
    "avalon_eval__zack_dreamer_new_z2r75675_final.pt__avalon_worlds__0824_full.tar.gz__final",
    "avalon_eval__zack_dreamer_new_385c6vbz_final.pt__avalon_worlds__0824_full.tar.gz__final",
    "avalon_eval__zack_dreamer_new_lb9b0hut_final.pt__avalon_worlds__0824_full.tar.gz__final",
    "avalon_eval__zack_dreamer_new_9o64zy4y_final.pt__avalon_worlds__0824_full.tar.gz__final",
    "avalon_eval__zack_dreamer_new_29rrbilj_final.pt__avalon_worlds__0824_full.tar.gz__final",
]
combined_dreamer_results = defaultdict(list)
raw_dreamer_datas = []
for result_key in result_keys:
    reported_data = json.loads(s3_client.load(result_key))
    raw_dreamer_datas.append(reported_data)
    missing_levels = valid_levels - set(reported_data["all_results"].keys())
    if len(missing_levels) > 0:
        logger.warning(f"{result_key} is missing {len(missing_levels)} levels, ignoring incomplete data")
    else:
        for k in valid_levels:
            level_data = reported_data["all_results"][k]
            combined_dreamer_results[k].append(get_human_normalized_score(k, level_data))


dreamer_score_numpys = task_dict_to_score_numpys(combined_dreamer_results)
logger.info("\nDreamer 50m aggregated scores")
print_aggregate_results(dreamer_score_numpys)

# %%

## TORCHBEAST CURRICULUM ABLATION

torchbeast_curriculum_results = {
    "500m": defaultdict(list),
    "no_curr": defaultdict(list),
    "multi_task_compositional": defaultdict(list),
    "multi_task_all": defaultdict(list),
}
for curriculum in torchbeast_curriculum_results.keys():
    for uuid in FINISHED_RUNS[curriculum]:
        result_key = get_wandb_result_key(uuid, FIXED_WORLD_KEY)
        reported_data = json.loads(s3_client.load(result_key))
        missing_levels = valid_levels - set(reported_data["all_results"].keys())
        if len(missing_levels) > 0:
            logger.warning(f"{result_key} is missing {len(missing_levels)} levels, ignoring incomplete data")
        else:
            for k in valid_levels:
                level_data = reported_data["all_results"][k]
                torchbeast_curriculum_results[curriculum][k].append(get_human_normalized_score(k, level_data))


curriciulum_score_numpys = {k: task_dict_to_score_numpys(v) for k, v in torchbeast_curriculum_results.items()}
logger.info("\nCurriculum ablation aggregated scores")
for curriculum, scores in curriciulum_score_numpys.items():
    logger.info(f'\nCurriculum: {curriculum.replace("_", " ").capitalize()}')
    print_aggregate_results(scores)


# %%

## TORCHBEAST SINGLE TASK ABLATIONS

torchbeast_single_task_results = {
    k.split("_")[-1]: defaultdict(list) for k in FINISHED_RUNS.keys() if k.startswith("single_task")
}

for task in torchbeast_single_task_results.keys():
    for uuid in FINISHED_RUNS[f"single_task_{task}"]:
        result_key = get_wandb_result_key(uuid, FIXED_WORLD_KEY)
        try:
            reported_data = json.loads(s3_client.load(result_key))
        except KeyError:
            logger.warning(f"Missing result key {result_key}")
            continue
        missing_levels = valid_levels - set(reported_data["all_results"].keys())
        if len(missing_levels) > 0:
            logger.warning(f"{result_key} is missing {len(missing_levels)} levels, ignoring incomplete data")
        else:
            for k in valid_levels:
                level_data = reported_data["all_results"][k]
                torchbeast_single_task_results[task][k].append(get_human_normalized_score(k, level_data))

single_task_score_numpys = {k: task_dict_to_score_numpys(v) for k, v in torchbeast_single_task_results.items()}
# logger.info("\nCurriculum ablation aggregated scores")
# for task, scores in single_task_score_numpys.items():
#     logger.info(f"\nSingle task: {task}")
#     print_aggregate_results(scores)
# torchbeast_aggregated_single_task_results = defaultdict(list)
# for key in valid_levels:
#     task = folder_name_lookup[key].split("__")[0]
#     torchbeast_aggregated_single_task_results[key] = torchbeast_single_task_results[task][key][:3]
# aggregated_single_task_score_numpys = task_dict_to_score_numpys(torchbeast_aggregated_single_task_results)

aggregated_single_task_score_numpy = {k: v[k] for k, v in single_task_score_numpys.items()}
logger.info(f"\nSingle task aggregated:")
print_aggregate_results(aggregated_single_task_score_numpy)

# %%

tasks = [x.value.lower() for x in TASKS_BY_TASK_GROUP[AvalonTaskGroup.ALL]]
single_task_matrix = np.zeros((len(tasks), len(tasks)))
single_task_list_data = []
for i, train_task in enumerate(tasks):
    for j, test_task in enumerate(tasks):
        mean_performance = metrics.aggregate_mean(single_task_score_numpys[train_task][test_task])
        single_task_matrix[i][j] = mean_performance
        single_task_list_data.append((train_task, test_task, mean_performance))

df = pd.DataFrame(single_task_list_data, columns=["train_task", "test_task", "mean_performance"])

# %%

pivot_data = df.pivot("train_task", "test_task", "mean_performance")
fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(pivot_data, annot=True, fmt=".2f", ax=ax, cbar=False)
ax.set_title("Mean performance on each testing task for single task training")
ax.set_xlabel("Test task")
ax.set_ylabel("Train task")

plt.savefig("single_task_training_matrix.eps", format="eps")

# %%

## TORCHBEAST 500m learning curve
learning_curve_checkpoints = {
    "bd10b76d-ce5b-4e0a-8514-47d75e777ae6": [
        "model_step_3878400.tar",
        "model_step_50310400.tar",
        "model_step_100633600.tar",
        "model_step_150387200.tar",
        "model_step_199104000.tar",
        "model_step_250752000.tar",
        "model_step_301296000.tar",
        "model_step_348435200.tar",
        "model_step_399235200.tar",
        "model_step_451862400.tar",
        "final",
    ],
    "cdb06a43-4dfb-4d52-878d-8495075a4a38": [
        "model_step_5440000.tar",
        "model_step_48128000.tar",
        "model_step_101523200.tar",
        "model_step_149372800.tar",
        "model_step_202572800.tar",
        "model_step_250252800.tar",
        "model_step_301126400.tar",
        "model_step_349993600.tar",
        "model_step_398956800.tar",
        "model_step_450038400.tar",
        "final",
    ],
    "42c4f566-49d1-45c6-a4f2-1185ac49238c": [
        "model_step_4803200.tar",
        "model_step_49753600.tar",
        "model_step_101196800.tar",
        "model_step_152124800.tar",
        "model_step_198067200.tar",
        "model_step_251043200.tar",
        "model_step_299270400.tar",
        "model_step_352342400.tar",
        "model_step_399164800.tar",
        "model_step_450905600.tar",
        "final",
    ],
}

targets = [50_000_000 * i for i in range(11)]
learning_curve_results = {i: defaultdict(list) for i in targets}
learning_curve_actual_frames = defaultdict(list)
for uuid, checkpoint_filenames in learning_curve_checkpoints.items():
    for target, checkpoint_filename in zip(targets, checkpoint_filenames):
        result_key = get_wandb_result_key(uuid, FIXED_WORLD_KEY, checkpoint_filename)
        if checkpoint_filename == "final":
            learning_curve_actual_frames[target].append(target)
        else:
            learning_curve_actual_frames[target].append(int(checkpoint_filename.split("_")[-1].split(".")[0]))
        try:
            reported_data = json.loads(s3_client.load(result_key))
        except KeyError:
            logger.warning(f"Missing result key {result_key}")
            continue
        missing_levels = valid_levels - set(reported_data["all_results"].keys())
        if len(missing_levels) > 0:
            logger.warning(f"{result_key} is missing {len(missing_levels)} levels, ignoring incomplete data")
        else:
            for k in valid_levels:
                level_data = reported_data["all_results"][k]
                learning_curve_results[target][k].append(get_human_normalized_score(k, level_data))

learning_curve_score_numpys = {k: task_dict_to_score_numpys(v) for k, v in learning_curve_results.items()}
logger.info("\nLearning curve scores")
for target, scores in learning_curve_score_numpys.items():
    logger.info(f"\nSteps: {target}")
    print_aggregate_results(scores)


# %%


def make_rows(latex_results: Dict, result_key: str, header: List):
    rows = [header]
    num_rows = len(PRINT_TASK_ORDER)
    for i in range(num_rows):
        task = PRINT_TASK_ORDER[i]
        task_text = task.lower()
        if task_text == "simple":
            task_text = "all basic"
        elif task_text == "compositional":
            task_text = "all comp."
        row = ["\\texttt{" + task_text + "}"]
        for key in latex_results:
            row.append(latex_results[key][task][result_key])
        rows.append(row)
    return rows


def drop_rows_by_columns(rows: List, ignore: List[int]):
    rows_without_ignored_columns = []
    for row in rows:
        new_row = []
        for i in range(len(row)):
            if i in ignore:
                continue
            new_row.append(row[i])
        rows_without_ignored_columns.append(new_row)
    return rows_without_ignored_columns


def print_latex_table(rows):
    table = tabulate(rows, headers="firstrow", tablefmt="latex_raw")
    table = table.replace("\\hline", "\\midrule")
    # manual fixes to get the right formatting
    fixes = {
        1: {"text": "\\toprule", "replace": True},
        20: {"text": "\\midrule", "replace": False},
        24: {"text": "\\midrule", "replace": False},
        27: {"text": "\\bottomrule", "replace": True},
    }
    for i, line in enumerate(table.split("\n")):
        if fixes.get(i):
            logger.info(fixes[i]["text"])
            if fixes[i].get("replace"):
                continue
        logger.info(line)


header = [
    "\\textbf{Task}",
    "\\textbf{PPO}",
    "\\textbf{Dreamer}",
    "\\textbf{IMPALA}",
    "\\textbf{IMPALA (500m)}",
    "\\textbf{IMPALA (no curriculum)}",
]

latex_results = {
    "ppo": latex_aggregate_results(ppo_score_numpys),
    "dreamer": latex_aggregate_results(dreamer_score_numpys),
    "impala": latex_aggregate_results(torchbeast_score_numpys),
    "impala_500m": latex_aggregate_results(curriciulum_score_numpys["500m"]),
    "impala_no_curr": latex_aggregate_results(curriciulum_score_numpys["no_curr"]),
}

rows = make_rows(latex_results, "mean", header)
print_latex_table(rows)


# %%


rows = make_rows(latex_results, "optimality_gap", header)
print_latex_table(rows)

# %%


header = [
    "\\textbf{Task}",
    "\\textbf{MT-TB}",
    "\\textbf{MT-TA}",
    "\\textbf{MT-TC}",
    "\\textbf{ST-B}",
]

latex_results = {
    "MT-TB": latex_aggregate_results(torchbeast_score_numpys),
    "MT-TA": latex_aggregate_results(curriciulum_score_numpys["multi_task_all"]),
    "MT-TC": latex_aggregate_results(curriciulum_score_numpys["multi_task_compositional"]),
    "ST-B": latex_aggregate_results(aggregated_single_task_score_numpy),
}

rows = make_rows(latex_results, "mean", header)
print_latex_table(rows)

# %%


rows = make_rows(latex_results, "optimality_gap", header)
print_latex_table(rows)

# %%

# CDF score plots

rcParams["font.family"] = "serif"
rcParams["mathtext.default"] = "regular"
plt.rc("text", usetex=False)


def get_cdf(xs, a=0, b=10, n_bins=1000):
    counts, bin_edges = np.histogram(xs, bins=np.linspace(a, b, n_bins + 1))
    counts = counts / counts.sum()
    cdf = counts[::-1].cumsum()[::-1]
    # array of evenly-spaced scores for the x coords of the plot
    bin_edges = bin_edges[:-1]
    return bin_edges, cdf


def get_tasks_in_sorted_order(data: Dict[str, np.ndarray]) -> List[str]:
    tasks = list(data.keys())
    tasks.sort(key=lambda x: get_cdf(data[x].flatten())[1][50], reverse=True)
    return [t for t in tasks if t not in {"simple", "compositional", "all"}]


def make_taskwise_cdf_plot(ax, data, title="", no_ylabel=False, task_order=None):
    if task_order is None:
        tasks = get_tasks_in_sorted_order(data)
    else:
        tasks = task_order

    bins, _ = get_cdf(data["eat"].flatten())
    cdfs = {}
    for task in tasks:
        _, cdfs[task] = get_cdf(data[task].flatten())

    colors = [
        hsv_to_rgb(c)
        for c in [
            (0.95, 0.7, 0.9),
            (0.1, 0.8, 0.9),
            (0.45, 0.8, 0.7),
            (0.55, 0.7, 0.9),
            (0.65, 0.75, 0.9),
            (0.75, 0.5, 0.9),
            (0, 0.1, 0.5),
        ]
    ]

    for i, task in enumerate(tasks):
        ax.plot(
            bins,
            cdfs[task],
            color=colors[i // 3],
            linestyle=["solid", "dashed", "dotted"][i % 3],
            lw=3,
            label=task,
            zorder=-i,
        )  # f'$\\texttt{{{task}}}$')

    ax.legend(ncol=2, loc=(0.53, 0.425), prop={"family": "monospace"})
    # plt.setp(ax.legend().texts, family='typewriter')

    ax.set_xlim((-0.05, 2))
    ax.set_ylim((-0.03, 1.03))
    ax.grid(linestyle="dotted", linewidth="0.5", color=(0.5, 0.5, 0.5))

    fs = 17
    ax.tick_params(labelsize=fs)
    ax.set_xlabel("Human-normalized score $\\tau$", fontsize=fs)
    ax.set_ylabel("Fraction of runs with score $\\geq \\tau$", fontsize=fs) if not no_ylabel else None
    ax.set_yticklabels([]) if no_ylabel else None
    ax.set_title(title, fontsize=fs, pad=15)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def make_aggregate_cdf_plot(ax, datasets, names, title="", no_ylabel=False):
    # colors = get_colors(len(names))
    # colors = [.9,.2,0], [0,.5,.8]
    colors = hsv_to_rgb([0.95, 0.7, 0.9]), hsv_to_rgb([0.65, 0.75, 0.9]), hsv_to_rgb((0.45, 0.8, 0.7))

    bins, _ = get_cdf(datasets[0]["eat"].flatten())

    # for x in [0,.5,1,1.5,2]:
    #   ax.axvline(x=x, color=(.5,.5,.5), linestyle='dotted', lw=1, zorder=-1000)
    # for y in [0,.2,.4,.6,.8,1]:
    #   ax.axhline(y=y, color=(.5,.5,.5), linestyle='dotted', lw=1, zorder=-1000)

    # cdfs = {}
    for i, (data, name) in enumerate(zip(datasets, names)):
        for j, task in enumerate(["simple", "compositional", "all"]):
            _, cdf = get_cdf(datasets[i][task].flatten())
            ax.plot(
                bins,
                cdf,
                color=colors[i],
                label=f'{name} ({task if task != "simple" else "basic"})',
                lw=3,
                linestyle=["dotted", "dashed", "solid"][j],
                zorder=-i,
            )

    ax.legend(ncol=1, prop={"family": "monospace"})

    ax.set_xlim((-0.05, 2))
    ax.set_ylim((-0.03, 1.03))
    ax.grid(linestyle="dotted", linewidth="0.5", color=(0.5, 0.5, 0.5))

    fs = 17
    ax.tick_params(labelsize=fs)
    ax.set_xlabel("Human-normalized score $\\tau$", fontsize=fs)
    ax.set_ylabel("Fraction of runs with score $\geq \\tau$", fontsize=fs) if not no_ylabel else None
    ax.set_yticklabels([]) if no_ylabel else None
    ax.set_title(title, fontsize=fs, pad=15)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


# %%

rcParams["font.family"] = "serif"
rcParams["mathtext.default"] = "regular"
plt.rc("text", usetex=False)

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 10))

task_order = get_tasks_in_sorted_order(torchbeast_score_numpys)
make_taskwise_cdf_plot(axs[0][0], torchbeast_score_numpys, title="IMPALA", task_order=task_order)
make_taskwise_cdf_plot(axs[0][1], ppo_score_numpys, title="PPO", no_ylabel=True, task_order=task_order)
make_taskwise_cdf_plot(axs[1][0], dreamer_score_numpys, title="Dreamer", no_ylabel=True, task_order=task_order)
make_aggregate_cdf_plot(
    axs[1][1],
    [torchbeast_score_numpys, ppo_score_numpys, dreamer_score_numpys],
    names=["IMPALA", "PPO", "Dreamer"],
    title="Aggregate",
    no_ylabel=True,
)

fig.tight_layout()

plt.savefig("taskwise_agent_scores.eps", format="eps")
# plt.savefig("taskwise_agent_scores.pdf")

# %%

ts = sorted(learning_curve_score_numpys.keys())
true_ts = [np.mean(learning_curve_actual_frames[k]) for k in ts]
tasks = [t for t in learning_curve_score_numpys[ts[0]].keys() if t not in ["simple", "compositional", "all"]]
curves = {task: [learning_curve_score_numpys[t][task].mean() for t in ts] for task in tasks}

# order tasks
# fig, ax = plt.subplots(figsize=(8, 6))


def plot_all_tasks(ax):
    colors = [
        hsv_to_rgb(c)
        for c in [
            (0.95, 0.7, 0.9),
            (0.1, 0.8, 0.9),
            (0.45, 0.8, 0.7),
            (0.55, 0.7, 0.9),
            (0.65, 0.75, 0.9),
            (0.75, 0.5, 0.9),
            (0, 0.1, 0.5),
        ]
    ]
    for i, task in enumerate(task_order):
        ax.plot(
            true_ts,
            curves[task],
            color=colors[i // 3],
            linestyle=["solid", "dashed", "dotted"][i % 3],
            lw=3,
            label=task,
            zorder=-i,
        )

    ax.legend(ncol=3, prop={"family": "monospace"})

    # ax.set_xlim((-.05,2))
    ax.set_ylim((-0.03, 1.03))
    ax.grid(linestyle="dotted", linewidth="0.5", color=(0.5, 0.5, 0.5))
    fs = 17
    ax.tick_params(labelsize=fs)
    ax.set_xlabel("Environment steps", fontsize=fs)
    ax.set_ylabel("Mean performance on task", fontsize=fs)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


fig, ax = plt.subplots(figsize=(8, 5))
plot_all_tasks(ax)
plt.savefig("taskwise_learning_curves.eps", format="eps")
