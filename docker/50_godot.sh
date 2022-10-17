#!/bin/bash
set -e
set -u
set -x

# TODO: update to 3.4.4.7
wget -nv https://github.com/Avalon-Benchmark/godot/releases/download/3.4.4.avalon.0.4/godot.egl.opt.64        -O /usr/local/bin/godot-3.4.4-egl-fast
wget -nv https://github.com/Avalon-Benchmark/godot/releases/download/3.4.4.avalon.0.4/godot.egl.opt.tools.64  -O /usr/local/bin/godot-3.4.4-egl-safe

chmod +x /usr/local/bin/godot-*

# NOTE: using the safe binary for easier debugging
ln -s /usr/local/bin/godot-3.4.4-egl-safe /usr/local/bin/godot

rm -rf /tmp/*
