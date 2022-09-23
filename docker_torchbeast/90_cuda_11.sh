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


apt-get install cuda-compiler-11-3 cuda-libraries-dev-11-3 cuda-nvml-dev-11-3 cuda-nvtx-11-3
apt-get clean

update-alternatives --install /usr/local/cuda cuda /usr/local/cuda-11.3 113
