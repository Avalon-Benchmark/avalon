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

# NOTE: this should be empty 98% of the time - use it exploratory Docker changes that aren't in their own script *yet*


# Stuff to get atari dreamer repros to work - eventually we should move this into a dockerfile only used for repros.
sudo -iu user pip install dm_control==1.0.5
sudo -iu user pip install protobuf==3.20.1
sudo -iu user pip install pyparsing==3.0.7
sudo -iu user pip install gym[atari,accept-rom-license,other]==0.25.2
sudo -iu user pip install setuptools==59.8.0
apt-get update
apt-get install ffmpeg cmake wget unrar libgl1-mesa-dev


apt-get clean
rm -rf /tmp/*
