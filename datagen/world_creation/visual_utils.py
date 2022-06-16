from pathlib import Path

from datagen.env_helper import create_env
from datagen.env_helper import create_vr_benchmark_config
from datagen.env_helper import get_null_vr_action
from datagen.godot_env import VRActionType


def visualize_worlds_in_folder(base_output_path: Path, resolution=1024, num_frames=20):

    world_paths = base_output_path.iterdir()
    env_seed = 0
    video_id = 0
    config = create_vr_benchmark_config()

    with config.mutable_clone() as config:
        config.recording_options.resolution_x = resolution
        config.recording_options.resolution_y = resolution
    action_type = VRActionType
    env = create_env(config, action_type)

    all_observations = []
    # if we want to take a few actions
    # null_action = get_null_vr_action()
    worlds_to_sort = []
    for world_path in world_paths:
        task, seed_str, difficulty_str = world_path.name.split("__")
        difficulty = float(difficulty_str.replace("_", "."))
        seed = int(seed_str)
        worlds_to_sort.append((task, difficulty, seed, world_path))

    for (task, difficulty, seed, world_path) in sorted(worlds_to_sort):
        print(f"Loading {world_path}")
        world_file = world_path / "main.tscn"
        observations = []
        observations.append(
            env.reset_nicely_with_specific_world(
                seed=env_seed,
                world_id=video_id,
                world_path=str(world_file),
            )
        )
        for i in range(num_frames):
            null_action = get_null_vr_action()
            obs, _ = env.act(null_action)
            observations.append(obs)

        all_observations.append(observations)

    return all_observations
