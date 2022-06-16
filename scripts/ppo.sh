#!/bin/bash
set -e
set -u

CODE=/opt/projects/avalon
DATA=/mnt/private

sudo mkdir -p -m 777 "${DATA}"

export PATH=/opt/venv/bin:$PATH
export PYTHONPATH=$CODE

exec python $CODE/agent/ppo/main.py
