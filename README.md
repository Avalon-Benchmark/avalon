# Status

This code is being made public so that it can be reviewed as part of our paper submission to the Neurips Datasets and Benchmarks track.

A future release (with additional docs, tests, and examples) will be made at a later date.

# Getting started

## Docker startup (recommended)

For running examples and training models,
the easiest way to get started is to create a docker container:

```bash
# build the docker image for PPO (https://openai.com/blog/openai-baselines-ppo/)
docker build -f ./docker/Dockerfile . --tag=avalon/ppo

# optionally, use the following to build an image that can be used to run torchbeast (https://github.com/facebookresearch/torchbeast)
docker build -f ./docker_torchbeast/Dockerfile . --tag=avalon/torchbeast

# start the docker container and forward ports for a jupyter notebook
# to enable wandb, add `-e WANDB_API_KEY=<your wandb key>`
docker run -it -p 8888:8888 -v $(pwd):/opt/projects/avalon --gpus 'all,"capabilities=compute,utility,graphics"' avalon/ppo
```

To start training PPO, run the following command:

```bash
# to enable wandb, add `-e WANDB_API_KEY=<your wandb key>`
docker run -it -p 8888:8888 -v $(pwd):/opt/projects/avalon --gpus 'all,"capabilities=compute,utility,graphics"' avalon/ppo ./scripts/ppo.sh
```

## Installing locally

Alternatively, the requirements in `docker/20_requirements.txt` and `docker/21_requirements.txt` can be manually installed locally:
```sh
pip install --no-cache --extra-index-url https://download.pytorch.org/whl/cu113 -r 20_requirements.txt -r 21_requirements.txt
```
You will also need to pull out the packages that are pip installed in `docker/41_pip_extras.sh`.


## Using avalon via the OpenAI Gym interface

Running Avalon is as simple as the following:
```python
# From notebooks/overview.ipynb
# Simple random sampling with the standard gym interface
from avalon.datagen.env_helper import create_env
from avalon.datagen.env_helper import create_vr_benchmark_config
from avalon.datagen.env_helper import display_video
from avalon.datagen.godot_env import VRActionType


config = create_vr_benchmark_config()
action_type = VRActionType
action_space = action_type.to_gym_space()
env = create_env(config, action_type)

observations = [env.reset()]

for _ in range(10):
    random_action = action_space.sample()
    observation, _reward, _is_done, _log = env.step(random_action)
    observations.append(observation)

display_video(observations)
```

For a full example on how to create random worlds, take actions as an agent, and display the resulting observations, see [gym_interface_example](./notebooks/gym_interface_example.sync.ipynb).

# Notebooks

**Using Avalon via the OpenAI Gym interface**

See [gym_interface_example](./notebooks/gym_interface_example.sync.ipynb) for an example of how to create random worlds, 
take actions as an agent, and display the resulting observations.

**World generation**

To generate a simple world see [create_simple_world](./notebooks/create_simple_world.sync.ipynb). 

To debug and evaluate generated worlds see [evaluate_worlds](./notebooks/evaluate_worlds.sync.ipynb).

**Evaluation**

[create_human_scores](./notebooks/create_human_scores.sync.ipynb) demonstrates how scores were calculated from the recorded
human runs. 

To reproduce our results from the paper using the model checkpoints, you can run [avalon_results](./notebooks/avalon_results.sync.ipynb). 
Note, you'll need a wandb api key in order to run this notebook. See [Getting started](#getting-started) above for more info.

**Building for VR**

To build a version of Avalon for the Oculus, see [create_oculus_build](./notebooks/create_oculus_build.sync.ipynb).

# Resources

* [Human scores](https://avalon-benchmark.s3.us-west-2.amazonaws.com/avalon__human_scores__935781fe-267d-4dcd-9698-714cc891e985.json)
* [Human observations](https://avalon-benchmark.s3.us-west-2.amazonaws.com/avalon__all_observations__935781fe-267d-4dcd-9698-714cc891e985.tar.gz)
* [Human normalized actions](https://avalon-benchmark.s3.us-west-2.amazonaws.com/avalon__all_actions__935781fe-267d-4dcd-9698-714cc891e985.tar.gz)
* [Human raw inputs](https://avalon-benchmark.s3.us-west-2.amazonaws.com/avalon__all_human_inputs__935781fe-267d-4dcd-9698-714cc891e985.tar.gz)
* [Evaluation worlds](https://avalon-benchmark.s3.us-west-2.amazonaws.com/avalon_worlds__2f788115-ea32-4041-8cae-6e7cd33091b7.tar.gz)
* [Custom Godot engine build](https://github.com/Avalon-Benchmark/godot/releases/)

All checkpoints are listed [here](./docs/checkpoints.md).
