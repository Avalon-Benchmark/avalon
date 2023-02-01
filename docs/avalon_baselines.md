
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

Logging uses [Weights and Biases](https://wandb.ai/site), so you'll need to create a (free) account to get logging.
Alternatively, remote collection can be disabled via the env variable `WANDB_MODE=offline` or `wandb offline`.

## Reproducing our paper results

The training commands above should replicate something very similar to what was run for our paper results. The hyperparameters should be the same as those used in the paper.

The commands above will train the model, and by default will evaluate it on a set of randomly generated evaluation worlds.
Checkpoints of the model should be automatically uploaded to Weights and Biases if configured.
In our paper, we evaluated the models on the same set of 1000 worlds that were used for our human play-tester baseline, in order to be able to directly compare human and agent performance on the exact same worlds.

Once you have a trained model checkpoint, you can validate it on the 
[set of evaluation worlds](https://avalon-benchmark.s3.us-west-2.amazonaws.com/avalon_worlds__benchmark_evaluation_worlds.tar.gz)
like so:
```sh
CHECKPOINT_ABSOLUTE_PATH=/path/to/checkpoint

# NOTE: If you move the worlds directory after downloading,
# or downloaded manually to a directory other than /tmp/avalon_worlds/benchmark_evaluation_worlds,
# you'll need to patch the references with `python -m avalon.for_humans patch_paths_in_evaluation_worlds` before running
EVAL_WORLDS_PATH=/tmp/avalon_worlds/benchmark_evaluation_worlds
python -m avalon.for_humans download_evaluation_worlds "$EVAL_WORLDS_PATH" --patch_path_references

python -m avalon.agent.train_ppo_avalon --is_training False --is_testing True \
  --resume_from "file://$CHECKPOINT_ABSOLUTE_PATH"  \
  --env_params.fixed_worlds_load_from_path "$EVAL_WORLDS_PATH"
```

The results will be saved to the wandb run.
The key `test/overall_success_rate` contains the summary metric that we reported in our paper, which is the percent of the worlds that the agent succeeded on (ie successfully at the food).
detailed metrics for the success of each task, and histograms of success by task and task difficulty are also presented in order to better understand the agent's capabilities.

If you find that our code isn't reproducing a result from our paper, please open an issue with details of what you've tried and we'll be happy to help figure out what's going on!

## Performance

- PPO: with a single 3090 GPU, we get ~900 env steps/sec with the default configuration (16 workers). See the metric `timings/cumulative_env_fps` in wandb.
- Dreamer: with 2 3090 GPUs, we get ~600 env steps/sec with the default configuration (16 workers). See the metric `timings/cumulative_env_fps
` in wandb.
