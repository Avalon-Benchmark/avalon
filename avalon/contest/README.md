# Contest setup

## Step 1: Install Avalon

See [installation instructions](../README.md).

## Step 2: Download the evaluation data

Download the evaluation data ([instructions](../docs/avalon_baselines.md)).

The link is: https://avalon-benchmark.s3.us-west-2.amazonaws.com/avalon_worlds__2f788115-ea32-4041-8cae-6e7cd33091b7.tar.gz

They must be located in `/tmp/avalon_worlds/2f788115-ea32-4041-8cae-6e7cd33091b7/`.

## Step 3: Run environment with a random model

```
PYTHONPATH=. python eval.py
```

## Step 4: Build docker container with agent

Build the Avalon Docker image:

```
docker build -f ./docker/Dockerfile . --target train --tag=avalon/train
```

Then, build the contest Docker image:

```
cd avalon/contest
docker build -f ./docker/Dockerfile . --tag=avalon_contest_agent
```

Verify that the agent runs:

```
docker run -it --gpus 'all,"capabilities=compute,utility,graphics"' avalon_contest_agent
```

## Step 5: Upload docker container to EvalAI for evaluation

See [instructions](https://cli.eval.ai/) for setting up EvalAI.

```
evalai push avalon_contest_agent:latest --phase random-dev-1882
```