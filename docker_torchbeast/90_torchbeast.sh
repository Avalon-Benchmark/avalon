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


apt-get install cmake

sudo -iu user pip install atari-py==0.2.9
sudo -iu user pip install gym[atari,accept-rom-license]==0.21.0 AutoROM.accept-rom-license==0.4.2 ale-py==0.7.5 autorom==0.4.2 importlib-resources==5.7.1
sudo -iu user pip install opencv-contrib-python-headless==4.6.0.66

sudo -iu user git clone --recurse-submodules --shallow-submodules --depth 2 --jobs 8 https://gitlab.com/generally-intelligent/torchbeast /tmp/torchbeast
sudo -iu user pip install /tmp/torchbeast/nest/

sudo -iu user bash -c 'cd /tmp/torchbeast && python ./setup.py install'

apt-get clean
rm -rf /tmp/*
