#!/bin/bash
set -e
set -u
set -x

sudo -iu user mkdir /home/user/.mujoco
sudo -iu user wget -nv https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
sudo -iu user tar -xzf mujoco210-linux-x86_64.tar.gz -C /home/user/.mujoco
sudo -iu user rm mujoco210-linux-x86_64.tar.gz
sudo -iu user wget -nv https://www.roboti.us/file/mjkey.txt -O /home/user/.mujoco/mjkey.txt

sudo -iu user pip install box2d-py==2.3.8
sudo -iu user pip install Shapely==1.7.0


# we're building the Docker image without a GPU, *and* we need to trick MuJoCo into compiling with EGL support
# ask bawr or bryden for the gory details, rough context is at:
# https://github.com/openai/mujoco-py/blob/v2.1.2.14/mujoco_py/builder.py#L26-L44
# https://github.com/openai/mujoco-py/blob/v2.1.2.14/mujoco_py/builder.py#L73-L79

echo "export LD_LIBRARY_PATH=/home/user/.mujoco/mujoco210/bin:/usr/local/nvidia/lib64" >> /home/user/.bashrc
mkdir -p /usr/local/nvidia/lib64
ln -s /usr/bin/true /usr/bin/nvidia-smi

# force mujoco to actually build
sudo -iu user python -c 'import mujoco_py.builder'

rm -f /usr/bin/nvidia-smi

# add missing py.typed until ray[rllib] permits 0.23.1
# https://github.com/ray-project/ray/blob/master/python/requirements.txt#L34
sudo -iu user touch "$(ls -d /opt/venv/lib/*/site-packages)/gym/py.typed"

rm -rf /tmp/*
