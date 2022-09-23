#!/bin/bash
set -e
set -u
set -x

sudo -iu user pip install --no-cache -r /tmp/20_requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113

rm -rf /tmp/*
