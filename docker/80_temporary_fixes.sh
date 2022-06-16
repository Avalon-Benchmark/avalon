#!/bin/bash
set -e
set -u
set -x

# NOTE: this should be empty 98% of the time - use it exploratory Docker changes that aren't in their own script *yet*

apt-get clean
rm -rf /tmp/*
