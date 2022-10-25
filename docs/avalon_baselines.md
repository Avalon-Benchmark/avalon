
We've developed a RL library as part of `avalon`, which includes fast and accurate PPO and [Dreamer (v2)](https://arxiv.org/abs/2010.02193) implementations. Check out [our research paper](https://openreview.net/pdf?id=TzNuIdrHoU) for a full explanation of these baselines and their performance on Avalon tasks.

Follow the Ubuntu [installing](../README.md#Installing) guide to set up your system for training, or use the Docker image. Training is only expected to work well on linux systems, as that's the only platform where our optimized headless Godot binary runs.

## Training

PPO:
```
python -m avalon.agent.train_ppo_avalon --project <your wandb project>
```

Dreamer:
```
python -m avalon.agent.train_dreamer_avalon --project <your wandb project>
```

Using the Docker image:
```sh
docker build -f ./docker/Dockerfile . --tag=avalon/training
docker run -it --gpus 'all,"capabilities=compute,utility,graphics"' -e WANDB_API_KEY=<your wandb key> avalon/ppo python -m avalon.agent.train_ppo_avalon --project <your wandb project>
```

These default configurations are very similar to the configurations used to generate our paper baseline results.

Training is highly configurable using command-line arguments; see eg `avalon/agent/train_ppo_avalon.py` as an example. Anything in the `AvalonPPOParams` class hierarchy can be passed as an argument; nested classes can be set using `.` syntax:

```
python -m avalon.agent.train_ppo_avalon --total_env_steps 5000000 env_params.training_protocol MULTI_TASK_EASY
```

Or get more frequent logging for debugging with something like this:
```
python -m avalon.agent.train_ppo_avalon --log_freq_hist 100 --log_freq_scalar 10 --log_freq_media 100
```

If you only have one GPU and want to train Dreamer (which defaults to using 2):
```
# Numbers are GPU ids (corresponding to nvidia-smi)
python -m avalon.agent.train_dreamer_avalon --train_gpu 0 --inference_gpus 0,
```

Logging uses [Weights and Biases](https://wandb.ai/site), so you'll need to create a (free) account to get logging (or you can disable it, with env variable `WANDB_MODE=offline`).

## Reproducing our paper results

The training commands above should replicate something very similar to what was run for our paper results. The hyperparameters should be the same as those used in the paper.

The commands above will train the model, and by default will evaluate it on a set of randomly generated evaluation worlds. Checkpoints of the model should be automatically uploaded to Weights and Biases. In our paper, we evaluated the models instead on the same set of 1000 worlds that were used for our human play-tester baseline, in order to be able to directly compare human and agent performance on the exact same worlds.

To run evaluation on this set of fixed worlds:
- download the [set of evaluation worlds](https://avalon-benchmark.s3.us-west-2.amazonaws.com/avalon_worlds__2f788115-ea32-4041-8cae-6e7cd33091b7.tar.gz) to `/mnt/private/avalon/worlds/viewable_worlds/` (it must be this path, as the world files contain absolute paths to resources that they'll try to load which are also in this folder).
- get the checkpoint of a trained model. you can use one of our [pretrained checkpoints](./docs/checkpoints.md), or your own trained model.
- run a command like this (exchanging `ppo` for `dreamer` or whatever algorithm your model was trained with):
```python -m avalon.agent.train_ppo_avalon --resume_from file://CHECKPOINT_ABS_PATH --is_training False --is_testing True --env_params.fixed_worlds_load_from_path /mnt/private/avalon/worlds/viewable_worlds/```
- the results will be in the wandb run. the key `test/overall_success_rate` contains the summary metric that we reported in our paper, which is the percent of the worlds that the agent succeeded on (ie successfully at the food). detailed metrics for the success of each task, and histograms of success by task and task difficulty are also presented in order to better understand the agent's capabilities.


If you find that our code isn't reproducing a result from our paper, please open an issue with details of what you've tried and we'll be happy to help figure out what's going on!

## Performance

- PPO: with a single 3090 GPU, we get ~900 env steps/sec with the default configuration (16 workers). See the metric `timings/cumulative_env_fps` in wandb.
- Dreamer: with 2 3090 GPUs, we get ~600 env steps/sec with the default configuration (16 workers). See the metric `timings/cumulative_env_fps
` in wandb.
