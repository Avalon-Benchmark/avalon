#!/bin/bash
set -e
set -u
set -x

# This replaces an install of libnvidia-gl, which won't work in a docker container

# beware, this is rather fiddly:
# - Docker is expected to start the container with --gpus='(...),capabilities=compute,utility,graphics"'
# - Docker automatically mounts the nvidia libraries from the host, pinning us to the correct version and bypassing apt
# - since we're not installing libnvidia-gl, we need to manually fix the config for EGL dispatch before we have library

rm  /usr/share/glvnd/egl_vendor.d/50_mesa.json
tee /usr/share/glvnd/egl_vendor.d/01_nvidia.json << EOF
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF
