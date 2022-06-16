#!/bin/bash

set -e
set -u

X=64
Y=128
F=10
D="${1%/}"

for N in $(seq 0 $2)
do

I="00000${N}"

A=(
    -v error
    -f rawvideo
    -pix_fmt rgb24
    -framerate "$F"
    -video_size "${X}x${Y}"
    -i $D/1/*/$I/rgb__byte_*.raw
    -f rawvideo
    -pix_fmt rgb24
    -framerate "$F"
    -video_size "${X}x${Y}"
    -i $D/2/*/$I/rgb__byte_*.raw
    -f rawvideo
    -pix_fmt rgb24
    -framerate "$F"
    -video_size "${X}x${Y}"
    -i $D/3/*/$I/rgb__byte_*.raw
    -f rawvideo
    -pix_fmt rgb24
    -framerate "$F"
    -video_size "${X}x${Y}"
    -i $D/4/*/$I/rgb__byte_*.raw
    -filter_complex hstack=inputs=4
    -y
    -framerate "$F"
    -fflags bitexact
    -vcodec libx264rgb
    -crf 0
    -movflags +faststart
    ${D}_${N}.mkv
)

ffmpeg "${A[@]}"

done
