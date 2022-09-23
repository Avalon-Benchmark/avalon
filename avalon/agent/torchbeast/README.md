# Running torchbeast

This folder contains the IMPALA implementation [TorchBeast](https://github.com/facebookresearch/torchbeast) from Facebook Research, with some changes to allow the Avalon benchmark to run properly. It is best run from a docker container as described in the main README.

To run the standard environment for 50m steps, just run:

```bash
python agent/torchbeast/polybeast.py
```

## Other configurations used in paper


To run for 500m steps:

```bash
python agent/torchbeast/polybeast.py --total_steps 500000000
```


To run on the Multi-Task All training protocol:

```bash
python agent/torchbeast/polybeast.py --training_protocol multi_task_all
```


To run on a single task training protocol:

```bash
python agent/torchbeast/polybeast.py --training_protocol single_task_eat
```

To run without the task curriculum,

```bash
python agent/torchbeast/polybeast.py --is_task_curriculum_disabled
```


## Evaluation

To evaluate a trained model, get the `[UUID]` generated for the run and call,

```bash
python agent/torchbeast/polybeast.py --mode test --fixed_world_path /tmp/science/avalon/worlds/viewable_worlds/ --fixed_world_key avalon_worlds__0824_full.tar.gz  --xpid [UUID]
```
