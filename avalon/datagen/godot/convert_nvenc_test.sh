#!/bin/bash

FILE='/tmp/godot/000000/rgb__byte_200_1024_512_3.raw'
FGBR='/tmp/godot/000000/gbr__byte_200_1024_512_3.raw'

ARGS=(
#   -n
    -y
    -v error
    -stats
    -f rawvideo
#   -pix_fmt yuv444p
    -framerate 10
    -video_size 512x1024
    -i "$FILE"
    -start_number 0
    -vsync 0
    -fflags bitexact
    -movflags +faststart
)

AGBR=(
#   -n
    -y
    -v error
    -stats
    -f rawvideo
#   -pix_fmt yuv444p
    -framerate 10
    -video_size 512x1024
    -i "$FGBR"
    -start_number 0
    -vsync 0
    -fflags bitexact
    -movflags +faststart
)

ffmpeg -pix_fmt yuv444p "${AGBR[@]}" -vcodec h264_nvenc -preset hq         'nvhq.gbr.mkv'
ffmpeg -pix_fmt yuv444p "${AGBR[@]}" -vcodec hevc_nvenc -preset losslesshp 'nvlh.gbr.mkv'
ffmpeg -pix_fmt yuv444p "${AGBR[@]}" -vcodec h264_nvenc -preset losslesshp 'nvll.gbr.mkv'
ffmpeg -pix_fmt yuv444p "${ARGS[@]}" -vcodec h264_nvenc -preset losslesshp 'nvll.yuv.mkv'
ffmpeg -pix_fmt yuv444p "${AGBR[@]}" -vcodec libx264    -crf 0             'x264.gbr.mkv'
ffmpeg -pix_fmt rgb24   "${ARGS[@]}" -vcodec libx264rgb -crf 0             'x264.rgb.mkv'
ffmpeg -pix_fmt yuv444p "${ARGS[@]}" -vcodec libx264    -crf 0             'x264.yuv.mkv'

echo

for MKV in *.mkv
do
    PIX=''
    case "$MKV"
    in
        *.gbr.mkv) PIX='yuv444p'
    ;;
        *.rgb.mkv) PIX='rgb24'
    ;;
        *.yuv.mkv) PIX='yuv444p'
    ;;
    esac
    ffmpeg -y -v error -i "$MKV" -f rawvideo -vcodec rawvideo -pix_fmt "$PIX" "${MKV/%mkv/raw}"
done

echo

md5sum *.raw

echo

md5sum "$FILE"
md5sum "$FGBR"
