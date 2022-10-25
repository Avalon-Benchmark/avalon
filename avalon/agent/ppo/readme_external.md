
This implementation is designed to closely match that in stable-baselines3, although it has been modified some since the original replication of those results.


## Launch commands 

See the `PPOParams` class (and parents) for all available flags. Logging is with weights and biases. You'll want to add the flag `--project YOUR_WANDB_PROJECT` to all of these.


To train (replicates paper training result):
`python -m agent.train_ppo_godot`

To evaluate on fixed human-baseline worlds (replicates paper evaluations):
`python -m agent.train_ppo_godot --is_training False --is_testing True --env_params.fixed_worlds_s3_key S3_KEY --resume_from wandb://WANDB_PROJECT/WANDB_RUN_ID/final.pt --env_params.fixed_worlds_load_from_path /tmp/science/avalon/worlds/viewable_worlds/`

