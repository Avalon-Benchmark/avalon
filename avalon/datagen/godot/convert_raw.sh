#!/bin/bash

set -e
set -u


FFPLAY=0
TO_MKV=0
TO_MP4=0
TO_PNG=0

MD5SUM=0

VF_STR=''
VIDEOS=()


if [[ ($# == 0) || ($1 == -h) || ($1 == --help) ]]
then
cat << EOF
usage: $0
       [-F,--ffplay[=FPS]]
       [-S,--size[=X[:Y]]]
       [-Z,--zoom[=X[:Y]]]
       [-T,--tile[=X[:Y]]]
       [-K,--to-mkv[=FPS]]
       [-M,--to-mp4[=FPS]]
       [-P,--to-png[=FPS]]
       [-D,--md5sum]
        RAW_PATH
       [RAW_PATH...]
where:
    --ffplay:   plays the raw file directly
    --size:     resize the output, defaults to 512
    --zoom:     resize the output, defaults to 4x
    --tile:     tiled grid output, defaults to 8:8
    --to-mkv:   create a lossless .mkv file
    --to-mp4:   create a workable .mp4 file
    --to-png:   create .png files, animated if FPS given
    --md5sum:   print the checksum for each video stream
    RAW_PATH... the files or directories to be converted
EOF
exit $(( $# == 0 ))
fi


while (( $# ))
do
    IFS='=:' read -r K V1 V2 V3 V4 <<< "$1"
    case "$K"
    in
        -F|--ffplay)
            FFPLAY="${V1:-10}"
    ;;
        -S|--size)
            X="${V1:-512}"
            Y="${V2:-$X}"
            (( (16 <= X) && (X <= 1920) ))
            (( (16 <= Y) && (Y <= 1920) ))
            VF_STR+=",scale=${X}:${Y}:flags=neighbor"
    ;;
        -Z|--zoom)
            X="${V1:-4}"
            Y="${V2:-$X}"
            (( (2 <= X) && (X <= 16) )) && X="${X}*iw"
            (( (2 <= Y) && (Y <= 16) )) && Y="${Y}*ih"
            VF_STR+=",scale=${X}:${Y}:flags=neighbor"
    ;;
        -T|--tile)
            X="${V1:-8}"
            Y="${V2:-$X}"
            (( (2 <= X) && (X <= 16) ))
            (( (2 <= Y) && (Y <= 16) ))
            VF_STR+=",tile=${X}x${Y}"
            if [[ $V3 ]]
            then
                VF_STR+=":margin=${V3}:padding=${V3}:color=${V4:-WhiteSmoke}"
            fi
    ;;
        -K|--to-mkv)
            TO_MKV="${V1:-10}"
    ;;
        -M|--to-mp4)
            TO_MP4="${V1:-10}"
    ;;
        -P|--to-png)
            TO_PNG="${V1:-1}"
    ;;
        -D|--md5sum)
            MD5SUM=1
    ;;
        -*)
            echo "unknown argument: '$1'"
            exit 2
    ;;
        *)
            if [[ "$1" == *__byte_*_*_*_*.raw ]]
            then
                VIDEOS+=("$1")
            else
                readarray -d '' -O "${#VIDEOS[@]}" -t VIDEOS < <(
                    find -L "$1" -name '*__byte_*_*_*_*.raw' -print0
                )
            fi
    ;;
    esac
    shift 1
done


_main() {
    A=()
    V=''
    N=0
    X=0
    Y=0
    Z=0
    for V in "${VIDEOS[@]}"
    do
        B="$(basename "${V}")"
        D="$(dirname  "${V}")"
        IFS='_.' read -r TYPE _NULL _BYTE N Y X Z _RAW <<< "$B"
        V="${V%%.raw}"
        ffrun MD5SUM
        ffrun FFPLAY
        ffrun TO_MKV
        ffrun TO_MP4
        ffrun TO_PNG
    done
}


ffrun() {
    T="${1}"
    F="${!1}"

    ffenc
    ffmd5
}


LQ_MP4=(
    -vcodec libx264
    -pix_fmt yuv420p
#   -pix_fmt yuv444p
    -crf 5
    -movflags +faststart
)

HQ_MKV=(
    -vcodec libx264rgb
    -crf 0
    -movflags +faststart
)


ffenc() {
    A=(
#       -v error
        -f rawvideo
        -pix_fmt rgba
        -framerate "$F"
        -video_size "${X}x${Y}"
        -i "${V}.raw"
#       -vf "null=${VF_STR}"
        -vf vflip,scale=1234:1370
        -y
        -start_number 0
        -vsync 0
        -fflags bitexact
    )

    echo "${A[@]}"

    case "${T}=${F}"
    in
        MD5SUM=*) E='md5'; A=()
    ;;
        FFPLAY=*) E='tmp'; A=(ffplay "${A[@]:0:14}" -autoexit)
    ;;
        TO_MKV=*) E='mkv'; A=(ffmpeg "${A[@]}" "${HQ_MKV[@]}" "${V}.${E}")
    ;;
        TO_MP4=*) E='mp4'; A=(ffmpeg "${A[@]}" "${LQ_MP4[@]}" "${V}.${E}")
    ;;
        TO_PNG=1) E='png'; A=(ffmpeg "${A[@]}" "${D}/${TYPE}__byte_%0${#N}d_${X}_${Y}_${Z}.${E}")
    ;;
        TO_PNG=*) E='png'; A=(ffmpeg "${A[@]}" -f "apng" "${V}.${E}")
    ;;
    esac

    if [[ $F != 0 ]]
    then
        find "$D" -maxdepth 1 -name "${TYPE}__*.${E}" -delete
        "${A[@]}"
    fi
}


ffmd5() {
    if [[ $MD5SUM == 0 || $F == 0 ]]
    then
        return 0
    fi

    case "${T}"
    in
        MD5SUM) R=''
    ;;
        FFPLAY) R=''
    ;;
        *) R="${A[-1]}"
    ;;
    esac

    A=(
        -v error
        -i "${R}"
        -f rawvideo
        -pix_fmt rgb24
        -start_number 0
        -
    )

    if [[ $R ]]
    then
        R="$(sed -r 's/_%(0[0-9]+)d_/_%.\1s_/' <<< "$R")"
        R="$(printf "${R}" '??????')"
        M="$(md5sum -b <(ffmpeg "${A[@]}"))"
    else
        R="${V}.raw"
        M="$(md5sum -b $R)"
    fi
    S="$(du -B1 -ach ${R} | tail -n 1 | cut -f 1)"
    echo -e "${M:0:32}\t${R}\t${S}"
}


_main
