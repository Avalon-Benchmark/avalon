
## DMC/Atari install

Run this to get the DMC/Atari environments.

```
pip install dm_control==1.0.5
pip install protobuf==3.20.1
pip install pyparsing==3.0.7
pip install gym[atari,accept-rom-license,other]==0.25.2
pip install setuptools==59.8.0

apt-get install ffmpeg cmake wget unrar libgl1-mesa-dev
```

## Launch commands

See the `DreamerParams` class (and parents) for all available flags. Logging is with weights and biases.
You'll want to add the flag `--project YOUR_WANDB_PROJECT` to all of these.

DMC proprio:
`python -m agent.train_dreamer_dmc_proprio`

DMC vision:
`MUJOCO_GL=egl python -m agent.train_dreamer_dmc_proprio --env_params.include_proprio False --env_params.include_rgb True`

Atari:
`python -m agent.train_dreamer_atari

Avalon (this should reproduce the main paper training (but not eval) with default hyperparams):
`python -m agent.train_dreamer_avalon`

Avalon easy:
`python -m agent.train_dreamer_avalon --env_params.task_difficulty_update 0 --env_params.training_protocol MULTI_TASK_EASY`

Avalon random world eval:
`python -m agent.train_dreamer_avalon --is_training False --is_testing True --resume_from wandb://WANDB_PROJECT/WANDB_RUN_ID/final.pt`

Avalon fixed human-baseline world eval (this is how we did the evaluations for the paper results):
`python -m agent.train_dreamer_avalon --is_training False --is_testing True --env_params.fixed_worlds_s3_key WORLD_KEY --resume_from wandb://WANDB_PROJECT/WANDB_RUN_ID/final.pt --env_params.fixed_worlds_load_from_path /tmp/science/avalon/worlds/viewable_worlds/`


## Implementation verification

We have verified this implementation by running all the deepmind-control proprioceptive and vision tasks, with 2 replicates, and comparing them to the same environments run on the official dreamerv2 repository. The training curves were very similar between the two implementations. The only change in hyperparameters from the official configuration is to not use discrete latents, which we haven't implemented and didn't find helpful in a quick test on deepmind-control. We performed a similar comparison on 10 Atari environments, also with continuous latents, with similarly good results.
