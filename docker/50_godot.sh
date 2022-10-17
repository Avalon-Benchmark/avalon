#!/bin/bash
set -e
set -u
set -x

pushd /tmp

wget -nv https://github.com/Avalon-Benchmark/godot/releases/download/3.4.4.avalon.0.9.0/linux-egl-runner.zip
wget -nv https://github.com/Avalon-Benchmark/godot/releases/download/3.4.4.avalon.0.9.0/linux-egl-editor.zip

unzip ./linux-egl-runner.zip
unzip ./linux-egl-editor.zip

mv -f ./godot.egl.opt.debug.64 /usr/local/bin/godot-runner-3.4.4-0.9.0
mv -f ./godot.egl.opt.tools.64 /usr/local/bin/godot-editor-3.4.4-0.9.0

chmod +x /usr/local/bin/godot-*

ln -s /usr/local/bin/godot-editor-3.4.4-0.9.0 /usr/local/bin/godot

popd

rm -rf /tmp/*
