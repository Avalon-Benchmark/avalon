#!/bin/bash
set -e
set -u
set -x

sudo -iu user pip install --no-cache -r /tmp/21_requirements.txt

sudo -iu user jupyter nbextension     install jupyter_ascending --py --sys-prefix
sudo -iu user jupyter nbextension     enable  jupyter_ascending --py --sys-prefix
sudo -iu user jupyter serverextension enable  jupyter_ascending --py --sys-prefix

rm -rf /tmp/*
