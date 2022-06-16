#!/bin/bash
set -e
set -u
set -x

sudo -iu user pip install --no-cache -r /tmp/20_requirements.txt -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

rm -rf /tmp/*
