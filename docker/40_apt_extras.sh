#!/bin/bash
set -e
set -u
set -x

function apt-get {
    set +x
    export DEBIAN_FRONTEND=noninteractive
    export DEBCONF_NONINTERACTIVE_SEEN=true
    if [[ $1 == 'install' ]]
    then
        /usr/bin/apt-get -s -qq "$@" | grep Inst | cut -d ' ' -f 1-2
        /usr/bin/apt-get -y -qq "$@" > /dev/null
    else
        /usr/bin/apt-get -y -qq "$@" > /dev/null
    fi
    set -x
}


apt-get update

apt-get install libegl-dev libopengl-dev --no-install-recommends

apt-get install libgeos-dev
apt-get install libglew-dev
apt-get install mesa-utils-extra

apt-get install build-essential
apt-get install patchelf
apt-get install swig

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

apt-get clean
rm -rf /tmp/*
