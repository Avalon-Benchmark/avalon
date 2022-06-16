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


apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
tee /etc/apt/sources.list.d/cuda.nvidia.list << EOF
deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /
EOF

apt-key adv --fetch-keys https://apt.kitware.com/keys/kitware-archive-latest.asc
tee /etc/apt/sources.list.d/cmake.kitware.list << EOF
deb https://apt.kitware.com/ubuntu/ focal main
EOF

apt-get update
apt-get clean
