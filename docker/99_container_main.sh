#!/bin/bash
set -e
set -u

CODE=/opt/projects/avalon
DATA=/mnt/private

mkdir -p ~/.ipython/profile_default/
cat > ~/.ipython/profile_default/ipython_config.py << EOF
c.InteractiveShellApp.exec_lines = ["import os; os.chdir('${CODE}')", "import avalon.common.hacks"]
EOF

sudo mkdir -m 777 "${DATA}"

# RL trash
export PYGLET_HEADLESS=True
export LD_LIBRARY_PATH=/home/user/.mujoco/mujoco210/bin:/usr/local/nvidia/lib64

# path fix
export PATH=/opt/venv/bin:$PATH
export PYTHONPATH=$CODE

# binding 0.0.0.0 because docker sometimes fails to forward ports on 127.0.0.1
exec jupyter-notebook --ip 0.0.0.0 --notebook-dir "${CODE}"
