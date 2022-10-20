#!/bin/bash
set -e
set -u
set -x

pushd /tmp

GODOT=3.4.4
PATCH=0.9.1

RUNNER=/usr/local/bin/godot-runner-$GODOT-$PATCH
EDITOR=/usr/local/bin/godot-editor-$GODOT-$PATCH

wget -nv https://github.com/Avalon-Benchmark/godot/releases/download/$GODOT.avalon.$PATCH/linux-egl-runner.zip
wget -nv https://github.com/Avalon-Benchmark/godot/releases/download/$GODOT.avalon.$PATCH/linux-egl-editor.zip

unzip ./linux-egl-runner.zip
unzip ./linux-egl-editor.zip

mv -f ./godot.egl.opt.debug.64 $RUNNER
mv -f ./godot.egl.opt.tools.64 $EDITOR

chmod +x /usr/local/bin/godot-*

ln -s $EDITOR /usr/local/bin/godot

popd

rm -rf /tmp/*.zip
