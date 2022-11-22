# Contest setup

## Step 1: Install Avalon

See [installation instructions](../README.md).

## Step 2: Download the evaluation data

Download the evaluation data:
* Minival phase: 
https://avalon-benchmark.s3.us-west-2.amazonaws.com/contest/minival-20221117-03e70547d557.tar.gz (20 worlds, 3.0 MB)
* Public test phase: https://avalon-benchmark.s3.us-west-2.amazonaws.com/contest/public-test-20221117-e3d7079216f4.tar.gz (1000 worlds, 1.0 GB)

Unzip them to `/tmp/avalon_worlds/minival/` and `/tmp/avalon_worlds/public_test/` (or anywhere else you prefer).

## Step 3: Run environment with a random model

Set the environment variable `FIXED_WORLDS_PATH` to the worlds you want to evaluate (eg: `/tmp/avalon_worlds/minival/`). Then run:

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

Follow [these instructions](https://cli.eval.ai/) for setting up EvalAI.

Run the following command to submit your container. The phase name must be one of: `avalon-minival`, `avalon-public-test`, or `avalon-private-test`.

```
evalai push avalon_contest_agent:latest --phase avalon-minival
```

Please make sure your container runs locally before submitting.