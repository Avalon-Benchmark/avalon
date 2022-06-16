#!/bin/bash
set -e
set -u
set -x

adduser --disabled-password --gecos 'User' --uid 1000 user
usermod --append --groups video,docker --comment User user

sudo -u user cp /tmp/bashrc /home/user/.bashrc
cp /tmp/htoprc /etc/

install -m 440 /dev/stdin /etc/sudoers.d/user << EOF
user ALL=(ALL:ALL) NOPASSWD: ALL
EOF

chown root:user /opt
chmod 775 /opt

sudo -iu user python3.9 -m venv /opt/venv
sudo -iu user pip install wheel

apt-get clean
rm -rf /tmp/*
