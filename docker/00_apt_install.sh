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


tee /etc/apt/sources.list << EOF
deb http://us.archive.ubuntu.com/ubuntu focal           main restricted universe multiverse
deb http://us.archive.ubuntu.com/ubuntu focal-updates   main restricted universe multiverse
deb http://us.archive.ubuntu.com/ubuntu focal-backports main restricted universe multiverse
deb http://security.ubuntu.com/ubuntu   focal-security  main restricted universe multiverse
EOF

apt-get update
apt-get install apt-utils debconf-utils
apt-get upgrade


debconf-set-selections << EOF
locales locales/default_environment_locale  select en_US.UTF-8
locales locales/locales_to_be_generated     select en_US.UTF-8 UTF-8
tzdata  tzdata/Areas                        select America
tzdata  tzdata/Zones/America                select Los_Angeles
EOF

apt-get install locales tzdata


apt-get install bash-completion
apt-get install ca-certificates
apt-get install dialog
apt-get install git
apt-get install gpgv
apt-get install htop
apt-get install iftop
apt-get install jq
apt-get install less
apt-get install moreutils
apt-get install nano
apt-get install ncdu
apt-get install pigz
apt-get install python3.9
apt-get install python3.9-dev
apt-get install python3.9-distutils
apt-get install python3.9-venv
apt-get install pv
apt-get install sudo
apt-get install tig
apt-get install tmux
apt-get install unzip
apt-get install vim
apt-get install wget
apt-get install xxd


apt-get install hwloc-nox pciutils
update-pciids -q

apt-get install openssh-server rsync --no-install-recommends
mkdir /run/sshd


addgroup --system --gid 200 docker
addgroup --system --gid 201 lxd

tee /etc/apt/sources.list.d/docker.list << EOF
deb https://download.docker.com/linux/ubuntu focal stable
EOF

wget -nv https://download.docker.com/linux/ubuntu/gpg -O /etc/apt/trusted.gpg.d/docker.asc

apt-get update
apt-get install docker-ce-cli


apt-get clean
rm -rf /tmp/*
