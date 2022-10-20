
We've developed a RL library as part of `avalon`, which includes fast and accurate PPO and [Dreamer (v2)](https://arxiv.org/abs/2010.02193) implementations. Check out [our research paper](https://openreview.net/pdf?id=TzNuIdrHoU) for a full explanation of these baselines and their performance on Avalon tasks.

Follow the Ubuntu [installing](../README.md#Installing) guide to set up your system for training, or use the Docker image.

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

## Performance

- PPO: with a single 3090 GPU, we get ~900 env steps/sec with the default configuration (16 workers). See the metric `timings/cumulative_env_fps` in wandb.
- Dreamer: with 2 3090 GPUs, we get ~600 env steps/sec with the default configuration (16 workers). See the metric `timings/cumulative_env_fps
` in wandb.
